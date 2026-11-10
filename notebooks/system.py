"""Pure composer of the Two-Step and One-Step systems (no gold labels, no training).

Public label contract (``public_labels``): the mapped top classes without ``other`` (taxonomy
order), then each parent's sub classes without ``other`` (parents in ``taxonomy.parents()``
order, classes in taxonomy order), then ``other``. Public ``other`` means "no mapped label"
(irrelevant or relevant-unmapped) and is exclusive.

Hard cascade (identical for both systems except the filter):

* Two-Step: a filter reject gives ``{other}`` and no categoriser output is used; One-Step has
  no filter and no relevance factor.
* Empty top prediction -> ``{other}`` (``empty_top_fallback``). ``other`` is dropped when any
  mapped top label is predicted (``other_conflict``).
* The sub categoriser of parent ``c`` runs once per batch on the rows whose PREDICTED top
  contains ``c``; its ``other`` is dropped (``sub_none_conflict`` when co-predicted with a real
  child); no real child keeps the parent (``parent_without_child``). Parents without a sub
  level (networking) pass through.

Components are adapters over deployed models. Each carries its embedder name; the systems get
a features provider ``features(embedder_name, row_ids) -> ndarray`` (rows in requested order).
Class mapping is by name, never by position. Heavy modules (``embeddings``, ``filtering``,
``categorisation``) are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from notebooks import taxonomy as _tx

Features = Callable[[str, Sequence[str]], np.ndarray]


class SystemCompositionError(Exception):
    pass


# --------------------------------------------------------------------------- labels and gold
def public_labels(taxonomy: Optional[_tx.Taxonomy] = None) -> List[str]:
    tx = taxonomy or _tx.default_taxonomy()
    other = tx.other
    labels = [c for c in tx.classes(_tx.TOP) if c != other]
    for parent in tx.parents():
        labels += [c for c in tx.classes(parent) if c != other]
    labels.append(other)
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise SystemCompositionError(f"public label names collide: {dup}")
    return labels


def gold_public(watson_list: Iterable[str], relevant: Any, taxonomy: Optional[_tx.Taxonomy] = None) -> List[str]:
    """Gold public labels in ``public_labels`` order."""
    tx = taxonomy or _tx.default_taxonomy()
    order = public_labels(tx)
    if not _tx.as_bool(relevant):
        return [tx.other]
    wl = list(watson_list)
    tops = [t for t in tx.top_labels(wl, True, "two_step") if t != tx.other]
    got = set(tops)
    for parent in tops:
        if parent in tx.sub_maps:
            got.update(s for s in tx.sub_labels(wl, parent) if s != tx.other)
    if not got:
        return [tx.other]
    return [x for x in order if x in got]


def gold_matrix(watson_lists: Iterable[Iterable[str]], relevant: Iterable[Any],
                taxonomy: Optional[_tx.Taxonomy] = None) -> np.ndarray:
    """Public gold binary matrix (n, len(public_labels)) from node-name lists and relevance."""
    tx = taxonomy or _tx.default_taxonomy()
    index = {c: j for j, c in enumerate(public_labels(tx))}
    rows = [gold_public(w, r, tx) for w, r in zip(watson_lists, relevant, strict=True)]
    out = np.zeros((len(rows), len(index)), dtype=np.int64)
    for i, labs in enumerate(rows):
        out[i, [index[x] for x in labs]] = 1
    return out


def gold_matrix_frame(df, watson_col: str = "watson", relevant_col: str = "relevant",
                      taxonomy: Optional[_tx.Taxonomy] = None) -> np.ndarray:
    """``gold_matrix`` from a DataFrame whose watson column holds the raw corpus field."""
    return gold_matrix([_tx.watson_list(w) for w in df[watson_col]], list(df[relevant_col]), taxonomy)


# --------------------------------------------------------------------------- adapters
def _as_name(meta: Optional[Mapping], fallback: Optional[str]):
    emb = (meta or {}).get("embedder")
    if isinstance(emb, Mapping):
        return emb.get("name") or fallback, emb.get("fingerprint")
    if isinstance(emb, str):
        return emb, None
    return fallback, None


class FilterAdapter:
    """Deployed filter (sklearn pipeline); the positive class is located via ``classes_``."""

    def __init__(self, model, embedder: str, fingerprint: Optional[Mapping] = None):
        classes = list(getattr(model, "classes_", []))
        pos = [j for j, c in enumerate(classes) if not isinstance(c, str) and c == 1]
        if len(pos) != 1:
            raise SystemCompositionError(f"filter classes_ {classes!r} do not contain exactly one positive (True/1) class")
        self.model, self.embedder, self.fingerprint = model, embedder, fingerprint
        self._pos, self._pos_value = pos[0], classes[pos[0]]

    def score(self, X) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=float)[:, self._pos]

    def accept(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X)) == self._pos_value


class CategoriserAdapter:
    """Deployed categoriser of one level, columns re-ordered to the taxonomy class order by name."""

    def __init__(self, categoriser, level: str, embedder: Optional[str] = None,
                 fingerprint: Optional[Mapping] = None, taxonomy: Optional[_tx.Taxonomy] = None,
                 source_level: Optional[str] = None):
        """``level`` is the taxonomy level (``top`` or a parent); ``source_level`` the level the deployed
        categoriser was trained for (sidecar/pickle name, e.g. ``general`` or ``one step``; default ``level``)."""
        tx = taxonomy or _tx.default_taxonomy()
        source_level = level if source_level is None else source_level
        want = tx.classes(level)
        have = list(categoriser.classes)
        if len(have) != len(set(have)) or set(have) != set(want):
            raise SystemCompositionError(f"{level}: categoriser classes {have} differ from taxonomy classes {want}")
        have_level = getattr(categoriser, "level", source_level)
        if have_level != source_level:
            raise SystemCompositionError(f"categoriser level {have_level!r} != {source_level!r}")
        self.categoriser, self.level, self.classes = categoriser, level, tuple(want)
        self.source_level = source_level
        self.embedder = embedder or categoriser.embedder
        self.fingerprint = fingerprint
        self._cols = [have.index(c) for c in want]

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.categoriser.predict(X))[:, self._cols].astype(np.int64)

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.categoriser.predict_proba(X), dtype=float)[:, self._cols]


# --------------------------------------------------------------------------- features providers
class ArtifactFeatures:
    """Features from embedding artifacts (``embeddings.load_artifact``), one per embedder name."""

    def __init__(self, directory, corpus: str, fingerprints: Optional[Mapping[str, Mapping]] = None,
                 expected_row_ids: Optional[Iterable] = None, *, allow_missing_fingerprint: bool = False,
                 **load_kwargs):
        """Fails closed: an embedder without an expected fingerprint raises unless ``allow_missing_fingerprint``."""
        self.allow_missing_fingerprint = allow_missing_fingerprint
        self.directory, self.corpus = directory, corpus
        self.fingerprints = dict(fingerprints or {})
        self.expected_row_ids = None if expected_row_ids is None else list(expected_row_ids)
        self.load_kwargs = load_kwargs
        self._cache: Dict[str, Any] = {}

    def __call__(self, embedder: str, row_ids: Sequence[str]) -> np.ndarray:
        if embedder not in self._cache:
            if embedder not in self.fingerprints and not self.allow_missing_fingerprint:
                raise SystemCompositionError(f"no expected fingerprint for embedder {embedder!r}")
            from notebooks import embeddings as em
            self._cache[embedder] = em.load_artifact(
                self.directory, em.artifact_name(self.corpus, embedder), self.expected_row_ids,
                self.fingerprints.get(embedder), **self.load_kwargs)
        return self._cache[embedder].select(row_ids)


def artifact_features(system, directory, corpus: str = "corpus", row_ids: Optional[Iterable] = None,
                      **load_kwargs) -> ArtifactFeatures:
    """``ArtifactFeatures`` wired with the fingerprints of ``system`` (see ``_Base.fingerprints``)."""
    return ArtifactFeatures(directory, corpus, system.fingerprints, row_ids, **load_kwargs)


class TextFeatures:
    """Features embedded on the fly from raw texts: ``texts`` maps row id -> text."""

    def __init__(self, texts: Mapping[str, str], embedders: Mapping[str, Any]):
        self.texts, self.embedders = dict(texts), dict(embedders)

    def __call__(self, embedder: str, row_ids: Sequence[str]) -> np.ndarray:
        from notebooks import embeddings as em
        if embedder not in self.embedders:
            raise SystemCompositionError(f"no embedder configured for {embedder!r}")
        return em.embed([self.texts[r] for r in row_ids], self.embedders[embedder])


# --------------------------------------------------------------------------- result
def _ro(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _frozen(d: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return MappingProxyType({k: _ro(v) for k, v in d.items()})


@dataclass(frozen=True)
class SystemResult:
    """Immutable composer output. Scores are NaN where the stage did not run; raw predictions are 0 there."""
    system: str
    row_ids: tuple
    labels: tuple
    P: np.ndarray
    filter_score: np.ndarray
    top_classes: tuple
    top_pred: np.ndarray
    top_scores: np.ndarray
    sub_classes: Mapping[str, tuple]
    sub_pred: Mapping[str, np.ndarray]
    sub_scores: Mapping[str, np.ndarray]
    sub_applied: Mapping[str, np.ndarray]
    filter_reject: np.ndarray
    top_other: np.ndarray
    empty_top_fallback: np.ndarray
    other_conflict: np.ndarray
    sub_none_conflict: Mapping[str, np.ndarray]
    parent_without_child: Mapping[str, np.ndarray]

    def routing_counts(self) -> Dict[str, int]:
        out = {k: int(getattr(self, k).sum()) for k in
               ("filter_reject", "top_other", "empty_top_fallback", "other_conflict")}
        out["sub_none_conflict"] = {p: int(v.sum()) for p, v in self.sub_none_conflict.items()}
        out["parent_without_child"] = {p: int(v.sum()) for p, v in self.parent_without_child.items()}
        return out


TWO_STEP_LEVEL, ONE_STEP_LEVEL = "general", "one step"   # deployed level names of the top categorisers

# --------------------------------------------------------------------------- systems
class _Base:
    name = ""

    def __init__(self, top: CategoriserAdapter, subs: Mapping[str, CategoriserAdapter],
                 taxonomy: Optional[_tx.Taxonomy] = None):
        self.taxonomy = tx = taxonomy or _tx.default_taxonomy()
        if top.level != _tx.TOP:
            raise SystemCompositionError(f"top adapter has level {top.level!r}")
        if set(subs) != set(tx.parents()):
            raise SystemCompositionError(f"sub categorisers {sorted(subs)} != parents {tx.parents()}")
        for p, a in subs.items():
            if a.level != p:
                raise SystemCompositionError(f"sub adapter for {p!r} has level {a.level!r}")
        self.top, self.subs = top, dict(subs)
        self.labels = public_labels(tx)
        self._index = {c: j for j, c in enumerate(self.labels)}

    def _components(self) -> List[Any]:
        return [self.top, *self.subs.values()]

    @property
    def embedders(self) -> List[str]:
        return sorted({c.embedder for c in self._components()})

    @property
    def fingerprints(self) -> Dict[str, Mapping]:
        """Embedder name -> fingerprint from the component metadata (where known)."""
        out: Dict[str, Mapping] = {}
        for c in self._components():
            if c.fingerprint is None:
                continue
            if out.setdefault(c.embedder, c.fingerprint) != c.fingerprint:
                raise SystemCompositionError(f"components disagree on the fingerprint of embedder {c.embedder!r}")
        return out

    def _run(self, row_ids: Sequence[str], features: Features, accept: Optional[np.ndarray],
             fscore: np.ndarray) -> SystemResult:
        tx, other = self.taxonomy, self.taxonomy.other
        ids = [str(r) for r in row_ids]
        n = len(ids)
        accept = np.ones(n, dtype=bool) if accept is None else accept
        tcls = self.top.classes
        top_pred = np.zeros((n, len(tcls)), dtype=np.int64)
        top_scores = np.full((n, len(tcls)), np.nan)
        if accept.any():
            sel = [i for i in range(n) if accept[i]]
            X = np.asarray(features(self.top.embedder, [ids[i] for i in sel]))
            top_pred[sel] = self.top.predict(X)
            top_scores[sel] = self.top.predict_proba(X)
        oi = tcls.index(other)
        mapped_cols = [j for j, c in enumerate(tcls) if c != other]
        any_mapped = top_pred[:, mapped_cols].sum(axis=1) > 0
        top_other = accept & (top_pred[:, oi] > 0)
        empty = accept & (top_pred.sum(axis=1) == 0)
        other_conflict = top_other & any_mapped

        P = np.zeros((n, len(self.labels)), dtype=np.int64)
        for j in mapped_cols:
            P[:, self._index[tcls[j]]] = top_pred[:, j] * accept
        sub_cls, sub_pred, sub_scores, applied, none_c, no_child = {}, {}, {}, {}, {}, {}
        for parent in tx.parents():
            ad = self.subs[parent]
            cls = ad.classes
            gate = accept & (top_pred[:, tcls.index(parent)] > 0)
            pred = np.zeros((n, len(cls)), dtype=np.int64)
            sc = np.full((n, len(cls)), np.nan)
            if gate.any():
                sel = np.flatnonzero(gate)
                Xs = np.asarray(features(ad.embedder, [ids[i] for i in sel]))
                pred[sel] = ad.predict(Xs)
                sc[sel] = ad.predict_proba(Xs)
            so = cls.index(other)
            real = [j for j, c in enumerate(cls) if c != other]
            has_child = pred[:, real].sum(axis=1) > 0
            for j in real:
                P[:, self._index[cls[j]]] = pred[:, j] * gate
            sub_cls[parent], sub_pred[parent], sub_scores[parent], applied[parent] = cls, pred, sc, gate
            none_c[parent] = gate & (pred[:, so] > 0) & has_child
            no_child[parent] = gate & ~has_child
        P[:, self._index[other]] = (P[:, [j for j in range(len(self.labels)) if self.labels[j] != other]]
                                    .sum(axis=1) == 0).astype(np.int64)
        return SystemResult(
            system=self.name, row_ids=tuple(ids), labels=tuple(self.labels), P=_ro(P),
            filter_score=_ro(fscore), top_classes=tuple(tcls), top_pred=_ro(top_pred), top_scores=_ro(top_scores),
            sub_classes=MappingProxyType(sub_cls), sub_pred=_frozen(sub_pred), sub_scores=_frozen(sub_scores),
            sub_applied=_frozen(applied), filter_reject=_ro(~accept if self.name == "two_step" else np.zeros(n, bool)),
            top_other=_ro(top_other), empty_top_fallback=_ro(empty), other_conflict=_ro(other_conflict),
            sub_none_conflict=_frozen(none_c), parent_without_child=_frozen(no_child))


class TwoStepSystem(_Base):
    name = "two_step"

    def __init__(self, filter_: FilterAdapter, top, subs, taxonomy=None):
        super().__init__(top, subs, taxonomy)
        self.filter = filter_

    def _components(self):
        return [self.filter, *super()._components()]

    def predict(self, row_ids: Sequence[str], features: Features) -> SystemResult:
        ids = [str(r) for r in row_ids]
        if not ids:
            return self._run(ids, features, np.zeros(0, bool), np.zeros(0))
        X = np.asarray(features(self.filter.embedder, ids))
        return self._run(ids, features, self.filter.accept(X), self.filter.score(X))


class OneStepSystem(_Base):
    name = "one_step"

    def predict(self, row_ids: Sequence[str], features: Features) -> SystemResult:
        ids = [str(r) for r in row_ids]
        return self._run(ids, features, None, np.full(len(ids), np.nan))


# --------------------------------------------------------------------------- loaders
ComponentHook = Callable[[str, Any, Mapping, Any], None]   # (component, path, sidecar metadata, adapter)


def _load_categorisers(paths: Mapping[str, Any], top_level: str, tx, allow_non_production: bool,
                       hook: Optional[ComponentHook] = None, subs: Optional[Mapping[str, CategoriserAdapter]] = None):
    """``paths`` is keyed by the real deployed level names (top: ``top_level``); ``subs`` reuses loaded sub adapters."""
    from notebooks import categorisation as ct
    need = [top_level] + ([] if subs is not None else list(tx.parents()))
    if set(paths) != set(need):
        raise SystemCompositionError(f"categoriser_paths levels {sorted(paths)} != required {sorted(need)}")
    out = {}
    for level in need:
        cat, meta = ct.load_deployed_categoriser(paths[level], level, allow_non_production=allow_non_production)
        name, fp = _as_name(meta, getattr(cat, "embedder", None))
        taxo = _tx.TOP if level == top_level else level
        out[level] = CategoriserAdapter(cat, taxo, name, fp, tx, source_level=level)
        if hook:
            hook(level, paths[level], meta, out[level])
    top = out.pop(top_level)
    return top, (dict(subs) if subs is not None else out)


def load_one_step(categoriser_paths: Mapping[str, Any], *, taxonomy=None, allow_non_production: bool = False,
                  hook: Optional[ComponentHook] = None, subs=None) -> OneStepSystem:
    """Paths keyed by real level names: top ``one step`` plus the taxonomy parents (omit them when ``subs`` given)."""
    tx = taxonomy or _tx.default_taxonomy()
    top, subs = _load_categorisers(categoriser_paths, ONE_STEP_LEVEL, tx, allow_non_production, hook, subs)
    return OneStepSystem(top, subs, tx)


def load_two_step(filter_path, categoriser_paths: Mapping[str, Any], *, taxonomy=None,
                  filter_embedder: Optional[str] = None, allow_non_production: bool = False,
                  hook: Optional[ComponentHook] = None, subs=None) -> TwoStepSystem:
    """Paths keyed by real level names: top ``general`` plus the taxonomy parents (omit them when ``subs`` given)."""
    from notebooks import filtering as fl
    tx = taxonomy or _tx.default_taxonomy()
    model, meta = fl.load_deployed_filter(filter_path)
    name, fp = _as_name(meta, filter_embedder)
    if not name:
        raise SystemCompositionError("filter sidecar names no embedder; pass filter_embedder")
    fad = FilterAdapter(model, name, fp)
    if hook:
        hook("filter", filter_path, meta, fad)
    top, subs = _load_categorisers(categoriser_paths, TWO_STEP_LEVEL, tx, allow_non_production, hook, subs)
    return TwoStepSystem(fad, top, subs, tx)
