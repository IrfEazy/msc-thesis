"""One frozen, grouped train/test split (with grouped inner folds) for all pipeline stages.

Every stage (filter, categorizers, System) uses the same split, so no post is seen in
training by one stage and in testing by another.

Grouping (``group_rows``): rows are keyed by ``textprep.group_key`` of their text, and keys
are then merged into components when one is a prefix of another (truncated copies such as
``"big breach at acme hits mill"`` vs ``"big breach at acme hits millions of users"``):
over the sorted distinct keys, A is linked to B whenever B starts with A and
``len(A) >= min_prefix``, for every such pair (full prefix closure, not only adjacent
keys), and links are closed transitively. The group is named after the smallest key of its
component. Over-merging only reduces the freedom of the stratification, it never leaks a
post across sides, so merging errs on the side of merging; ``max_group_rows`` makes a
runaway chain fail closed instead of silently swallowing a large part of the corpus.

``build_split`` assigns whole groups to test/train, then train groups to inner folds, with
iterative stratification on groups (Sechidis et al. 2011, generalised to groups weighted by
their row counts) over the label incidences returned by ``stratification_labels``:
relevance, two-step top-level labels of relevant rows and sub-level labels of relevant rows
under each of their top-level parents. Deterministic for a seed and independent of the row
order of the input frame.

The manifest is a DataFrame (one row per corpus row, in input order) with columns
``row_id`` (string form of ``id_col``, which must be unique), ``group`` (``"g"`` + 16 hex
chars of the sha256 of the component's smallest key), ``split`` (``"train"``/``"test"``)
and ``inner_fold`` (``0..inner_folds-1``, ``-1`` for test). Its metadata dict
(``manifest.attrs["metadata"]``) records the algorithm, every parameter, the corpus, input
and manifest digests, ``TEXTPREP_VERSION``, the taxonomy digest, grouping statistics and
counts; ``write_manifest``/``read_manifest`` store it as CSV plus a JSON sidecar, and
``verify_manifest`` checks a manifest against the corpus by recomputing the grouping and
rebuilding it. ``balance_report`` gives per-label support by split and inner fold.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from notebooks import taxonomy as tx
from notebooks.textprep import TEXTPREP_VERSION, group_key

ALGORITHM = "grouped-iterative-stratification"
SPLITS_VERSION = 2
MANIFEST_COLUMNS = ["row_id", "group", "split", "inner_fold"]
DEFAULT_COLUMNS = {"id": "_id", "text": "text", "relevant": "relevant", "watson": "watson"}
# Row order is not part of the corpus identity: records are hashed sorted by row id.
ROW_ORDER_POLICY = "sorted-by-row-id"


class ManifestError(AssertionError):
    """A manifest does not match the corpus, its metadata or a rebuild."""


def prefix_components(keys: Iterable[str], min_prefix: int) -> Dict[str, str]:
    """Map every distinct key to the smallest key of its prefix component.

    Over the sorted distinct keys, A is linked to B whenever B starts with A and
    ``len(A) >= min_prefix`` (all such pairs, e.g. ``a`` links both ``ab`` and ``ac``);
    components are the transitive closure. Keys sharing a prefix are contiguous in sorted
    order, so a stack of the current key's qualifying prefixes finds every link in
    O(n log n); the root of a component is a prefix of all its members, hence its smallest
    key."""
    if int(min_prefix) != min_prefix or min_prefix < 1:
        raise ValueError("min_prefix must be an int >= 1")
    root: Dict[str, str] = {}
    stack: List[str] = []
    for key in sorted(set(keys)):
        while stack and not key.startswith(stack[-1]):
            stack.pop()
        root[key] = root[stack[-1]] if stack else key
        if len(key) >= min_prefix:
            stack.append(key)
    return root


def group_rows(texts: Iterable[str], min_prefix: int) -> List[str]:
    """Group key of each row: the root (smallest key) of the prefix component of its
    ``group_key`` (see ``prefix_components``)."""
    keys = [group_key(t) for t in texts]
    root = prefix_components(keys, min_prefix)
    return [root[k] for k in keys]


def grouping_stats(texts: Iterable[str], min_prefix: int) -> Dict[str, Any]:
    """Statistics of the grouping: groups, merged components (groups made of more than one
    distinct key), distinct keys absorbed by merging, and the row-size distribution of
    merged components."""
    keys = [group_key(t) for t in texts]
    root = prefix_components(keys, min_prefix)
    n_keys = Counter(root.values())
    rows = Counter(root[k] for k in keys)
    merged = sorted((rows[r] for r, n in n_keys.items() if n > 1), reverse=True)
    return {
        "distinct_keys": len(root),
        "groups": len(n_keys),
        "merged_components": len(merged),
        "keys_absorbed": len(root) - len(n_keys),
        # JSON-safe: string keys ("rows in the component" -> number of components).
        "merged_component_rows": {str(n): c for n, c in sorted(Counter(merged).items())},
        "largest_merged_component_rows": merged[0] if merged else 0,
    }


def stratification_labels(
    relevant: Iterable[Any], watson_lists: Iterable[Sequence[str]], taxonomy: Optional[tx.Taxonomy] = None
) -> List[List[str]]:
    """Per-row stratification labels: ``relevant``/``irrelevant``; for relevant rows also
    ``top:<label>`` (two-step top labels, incl. ``top:other``) and, for each top label with a
    sub level, ``sub:<parent>:<label>`` (incl. the internal sub-level other)."""
    tax = tx._tax(taxonomy)
    out = []
    for rel, wl in zip(relevant, watson_lists):
        if not tx.as_bool(rel):
            out.append(["irrelevant"])
            continue
        top = tax.top_labels(wl, True, "two_step")
        labels = ["relevant"] + [f"top:{t}" for t in top]
        for parent in top:
            if parent in tax.sub_maps:
                labels += [f"sub:{parent}:{s}" for s in tax.sub_labels(wl, parent)]
        out.append(labels)
    return out


def _stratify(counts: np.ndarray, sizes: np.ndarray, fractions: Sequence[float], rng) -> np.ndarray:
    """Assign each group (row of ``counts``: label incidences, ``sizes``: rows) to a fold.

    Sechidis iterative stratification adapted to groups: repeatedly take the label held by
    the fewest unassigned groups and place its groups, largest first (seeded random order
    among equal sizes), each in the fold with the highest need, ties broken by the fold
    that most lacks rows, then randomly. Groups without labels are placed by rows only.

    Need is relative (remaining / target per fold and label, so 0.2/0.8 folds are treated
    alike; absolute deficits sent every large group to train) and is averaged over all the
    group's labels weighted by its incidences, not taken from the current label alone (a
    group also carries coarse labels such as relevance, and a single-label choice skewed
    them). Need is measured AFTER placing the group (remaining minus the group's incidences),
    so a group larger than a fold's remaining share of a label goes where it overshoots
    least instead of to a coin flip between equally needy folds.

    Corpus evidence (CyberPosts, label balance only, seeds 0-9): the single-label rule gave
    up to 8.6 points of test-share skew; need before placement, once truncated copies were
    merged into larger groups, sent a 44-row group holding 39% of sub-level
    "operating systems: other" to test (18.6 points) for 9 of 10 seeds; need after placement
    keeps every label within 2.8 points of the test fraction, with inner-fold deviations at
    the bound forced by the largest group of each label."""
    n_groups = len(sizes)
    k = len(fractions)
    frac = np.asarray(fractions, dtype=float)
    target_rows = frac * sizes.sum()
    target_lab = np.outer(frac, counts.sum(axis=0))
    want_rows = target_rows.copy()
    want_lab = target_lab.copy()
    safe_target = np.where(target_lab > 0, target_lab, 1.0)
    has = counts > 0
    priority = rng.permutation(n_groups)
    fold = np.full(n_groups, -1, dtype=int)
    remaining = np.ones(n_groups, dtype=bool)
    while remaining.any():
        n_rem = has[remaining].sum(axis=0)
        if n_rem.any():
            ties = np.flatnonzero(n_rem == n_rem[n_rem > 0].min())
            label = int(ties[rng.integers(len(ties))])
            members = np.flatnonzero(remaining & has[:, label])
        else:
            label = None
            members = np.flatnonzero(remaining)
        members = members[np.lexsort((priority[members], -sizes[members]))]
        for g in members:
            noise = rng.random(k)
            rows_need = want_rows / target_rows
            if label is None:
                f = int(np.lexsort((noise, -rows_need))[0])
            else:
                need = ((want_lab - counts[g]) / safe_target) @ (counts[g] / counts[g].sum())
                f = int(np.lexsort((noise, -rows_need, -need))[0])
            fold[g] = f
            remaining[g] = False
            want_rows[f] -= sizes[g]
            want_lab[f] -= counts[g]
    return fold


def _group_id(key: str) -> str:
    return "g" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _row_ids(df: pd.DataFrame, id_col: str) -> pd.Series:
    if id_col not in df.columns:
        raise ValueError(f"id column {id_col!r} not in frame")
    ids = df[id_col].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"id column {id_col!r} is not unique; pass a unique row id column")
    return ids.reset_index(drop=True)


def _raw_str(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def corpus_sha256(df: pd.DataFrame, columns: Optional[Dict[str, str]] = None) -> str:
    """Canonical digest of the data the split depends on: row id, raw text, relevance (as
    bool) and raw Watson field of every row, records sorted by row id
    (``ROW_ORDER_POLICY``), plus the column mapping. Independent of row order and of other
    columns."""
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    ids = _row_ids(df, cols["id"])
    records = sorted(
        zip(
            ids,
            df[cols["text"]].map(_raw_str),
            df[cols["relevant"]].map(lambda v: bool(tx.as_bool(v))),
            df[cols["watson"]].map(_raw_str),
        )
    )
    return _sha256_json({"row_order": ROW_ORDER_POLICY, "columns": cols, "records": records})


def _config(meta: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("algorithm", "splits_version", "seed", "test_fraction", "inner_folds", "min_prefix",
            "max_group_rows", "columns", "row_order", "textprep_version", "taxonomy_sha256")
    return {k: meta[k] for k in keys}


def input_sha256(corpus_digest: str, meta: Dict[str, Any]) -> str:
    """Digest of everything that determines the manifest: the corpus digest plus the split
    configuration (algorithm, version, seed, fractions, folds, grouping parameters, column
    mapping, row-order policy, TEXTPREP_VERSION and taxonomy digest)."""
    return _sha256_json({"corpus_sha256": corpus_digest, "config": _config(meta)})


def manifest_sha256(manifest: pd.DataFrame) -> str:
    """Canonical digest of a manifest: its four columns, rows sorted by ``row_id``, as CSV.
    Independent of the manifest's row order and of how it was stored."""
    m = manifest[MANIFEST_COLUMNS].astype({"row_id": str, "group": str, "split": str})
    m = m.astype({"inner_fold": int}).sort_values("row_id", kind="mergesort")
    data = m.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _counts(manifest: pd.DataFrame, inner_folds: int) -> Dict[str, Any]:
    sizes = manifest["group"].value_counts()
    split = manifest.drop_duplicates("group").set_index("group")["split"]
    train = manifest[manifest["split"] == "train"]
    return {
        "rows": int(len(manifest)),
        "groups": int(len(sizes)),
        "duplicate_groups": int((sizes > 1).sum()),
        "rows_in_duplicate_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "train_rows": int(len(train)),
        "test_rows": int((manifest["split"] == "test").sum()),
        "train_groups": int((split == "train").sum()),
        "test_groups": int((split == "test").sum()),
        "inner_fold_rows": [int((train["inner_fold"] == f).sum()) for f in range(inner_folds)],
        "inner_fold_groups": [
            int(train.loc[train["inner_fold"] == f, "group"].nunique()) for f in range(inner_folds)
        ],
    }


def build_split(
    df: pd.DataFrame,
    seed: int,
    test_fraction: float = 0.2,
    inner_folds: int = 5,
    *,
    min_prefix: int = 40,
    max_group_rows: Optional[int] = 300,
    columns: Optional[Dict[str, str]] = None,
    taxonomy: Optional[tx.Taxonomy] = None,
) -> pd.DataFrame:
    """Build the grouped, stratified split manifest of ``df`` (see module docstring).

    ``min_prefix`` is the minimum key length for prefix merging (``prefix_components``).
    ``max_group_rows`` caps the rows of any group (merged component or exact copies);
    a larger group raises ``ValueError`` (fail closed on runaway merges; ``None`` disables
    the cap). ``columns`` overrides the column names in ``DEFAULT_COLUMNS`` (keys ``id``,
    ``text``, ``relevant``, ``watson``). The id column must be unique (``ValueError``
    otherwise); in the CyberPosts corpus ``_id`` (Mongo id) is unique. Texts must be
    strings. Raises ``ValueError`` if the test split or any inner fold comes out empty.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if int(inner_folds) != inner_folds or inner_folds < 2:
        raise ValueError("inner_folds must be an int >= 2")
    if max_group_rows is not None and (int(max_group_rows) != max_group_rows or max_group_rows < 1):
        raise ValueError("max_group_rows must be None or an int >= 1")
    inner_folds = int(inner_folds)
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    tax = tx._tax(taxonomy)
    ids = _row_ids(df, cols["id"])
    texts = list(df[cols["text"]])
    keys = group_rows(texts, min_prefix)
    wls = [tx.watson_list(w) for w in df[cols["watson"]]]
    labels = stratification_labels(df[cols["relevant"]], wls, tax)

    # Groups sorted by key and labels sorted by name: independent of the input row order.
    uniq, inverse = np.unique(np.asarray(keys, dtype=object), return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(uniq)).astype(np.int64)
    if max_group_rows is not None and sizes.max() > max_group_rows:
        big = sorted(((int(n), str(k)[:60]) for k, n in zip(uniq, sizes) if n > max_group_rows), reverse=True)
        raise ValueError(f"{len(big)} group(s) exceed max_group_rows={max_group_rows}: {big[:5]}; "
            "pass max_group_rows=None or a larger value")
    label_names = sorted({label for row in labels for label in row})
    label_index = {name: i for i, name in enumerate(label_names)}
    counts = np.zeros((len(uniq), len(label_names)), dtype=np.int64)
    for g, row in zip(inverse, labels):
        for label in row:
            counts[g, label_index[label]] += 1

    rng = np.random.default_rng(int(seed))
    outer = _stratify(counts, sizes, [test_fraction, 1 - test_fraction], rng)
    if not (outer == 0).any() or not (outer == 1).any():
        raise ValueError("the test or train split is empty; the corpus has too few groups")
    group_fold = np.full(len(uniq), -1, dtype=int)
    train_groups = np.flatnonzero(outer == 1)
    group_fold[train_groups] = _stratify(
        counts[train_groups], sizes[train_groups], [1 / inner_folds] * inner_folds, rng
    )
    empty = [f for f in range(inner_folds) if not (group_fold[train_groups] == f).any()]
    if empty:
        raise ValueError(f"inner fold(s) {empty} are empty; too few train groups for {inner_folds} folds")

    gid = np.array([_group_id(k) for k in uniq], dtype=object)
    manifest = pd.DataFrame(
        {
            "row_id": ids,
            "group": gid[inverse],
            "split": np.where(outer[inverse] == 0, "test", "train"),
            "inner_fold": group_fold[inverse].astype(int),
        }
    )
    meta: Dict[str, Any] = {
        "algorithm": ALGORITHM,
        "splits_version": SPLITS_VERSION,
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "inner_folds": inner_folds,
        "min_prefix": int(min_prefix),
        "max_group_rows": None if max_group_rows is None else int(max_group_rows),
        "columns": cols,
        "row_order": ROW_ORDER_POLICY,
        "textprep_version": TEXTPREP_VERSION,
        "taxonomy_sha256": tax.sha256,
        "corpus_sha256": corpus_sha256(df, cols),
    }
    meta["input_sha256"] = input_sha256(meta["corpus_sha256"], meta)
    meta["manifest_sha256"] = manifest_sha256(manifest)
    meta["grouping"] = grouping_stats(texts, min_prefix)
    meta["counts"] = _counts(manifest, inner_folds)
    manifest.attrs["metadata"] = meta
    return manifest


def _metadata(manifest: pd.DataFrame, metadata: Optional[Dict]) -> Dict:
    meta = metadata if metadata is not None else manifest.attrs.get("metadata")
    if not meta:
        raise ManifestError("manifest has no metadata")
    return meta


def _sidecar(path: Union[str, Path]) -> Path:
    return Path(path).with_suffix(".json")


def write_manifest(manifest: pd.DataFrame, path: Union[str, Path], metadata: Optional[Dict] = None) -> Path:
    """Write the manifest CSV to ``path`` and its metadata (plus the sha256 of the CSV
    bytes) to the sidecar ``path`` with suffix ``.json``. Returns the sidecar path."""
    meta = dict(_metadata(manifest, metadata))
    data = manifest[MANIFEST_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    meta["manifest_csv_sha256"] = hashlib.sha256(data).hexdigest()
    Path(path).write_bytes(data)
    sidecar = _sidecar(path)
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def read_manifest(path: Union[str, Path]) -> pd.DataFrame:
    """Read a manifest written by ``write_manifest``; metadata goes to ``attrs["metadata"]``.
    Raises ``ManifestError`` if the CSV bytes do not match the recorded sha256."""
    data = Path(path).read_bytes()
    meta = json.loads(_sidecar(path).read_text(encoding="utf-8"))
    if hashlib.sha256(data).hexdigest() != meta.get("manifest_csv_sha256"):
        raise ManifestError(f"{path}: CSV sha256 does not match its metadata")
    manifest = pd.read_csv(
        path,
        dtype={"row_id": str, "group": str, "split": str, "inner_fold": int},
        keep_default_na=False,
    )
    manifest.attrs["metadata"] = meta
    return manifest


def verify_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    metadata: Optional[Dict] = None,
    taxonomy: Optional[tx.Taxonomy] = None,
    *,
    expected_seed: Optional[int] = None,
    expected_manifest_sha256: Optional[str] = None,
) -> None:
    """Raise ``ManifestError`` unless ``manifest`` is the split of ``df``.

    Checks: the recorded seed equals ``expected_seed`` and the manifest digest equals
    ``expected_manifest_sha256`` (when given; pin both in the consuming code so a swapped
    manifest is caught); every corpus row occurs exactly once; every row's group equals
    the group recomputed from the source texts with the recorded ``min_prefix``; split
    values are ``train``/``test``, test rows have inner fold -1 and train rows a fold in
    ``0..inner_folds-1``; no group is in two splits or two inner folds; the metadata
    (algorithm, versions, taxonomy, corpus, input and manifest digests, grouping stats,
    counts) matches; and rebuilding with the recorded parameters reproduces the manifest.
    """
    meta = _metadata(manifest, metadata)
    if expected_seed is not None and meta.get("seed") != expected_seed:
        raise ManifestError(f"manifest seed {meta.get('seed')!r} != expected seed {expected_seed!r}")
    if list(manifest.columns[: len(MANIFEST_COLUMNS)]) != MANIFEST_COLUMNS:
        raise ManifestError(f"manifest columns must start with {MANIFEST_COLUMNS}")
    if not pd.api.types.is_integer_dtype(manifest["inner_fold"]):
        raise ManifestError("inner_fold must be integer")
    if expected_manifest_sha256 is not None and manifest_sha256(manifest) != expected_manifest_sha256:
        raise ManifestError("manifest sha256 does not match the expected digest")
    cols = {**DEFAULT_COLUMNS, **meta.get("columns", {})}
    tax = tx._tax(taxonomy)
    ids = _row_ids(df, cols["id"])
    m_ids = manifest["row_id"].astype(str)
    if m_ids.duplicated().any():
        raise ManifestError("a row id occurs more than once in the manifest")
    if set(m_ids) != set(ids):
        raise ManifestError("manifest rows differ from corpus rows")
    m = manifest.assign(row_id=m_ids).set_index("row_id").loc[ids]
    texts = list(df[cols["text"]])
    min_prefix = meta.get("min_prefix")
    if not isinstance(min_prefix, int):
        raise ManifestError("metadata has no integer min_prefix")
    expected_group = np.array([_group_id(k) for k in group_rows(texts, min_prefix)], dtype=object)
    wrong = int((m["group"].to_numpy(dtype=object) != expected_group).sum())
    if wrong:
        raise ManifestError(f"{wrong} rows have a group that differs from the recomputed grouping")
    if not m["split"].isin(["train", "test"]).all():
        raise ManifestError("split values must be 'train' or 'test'")
    if m.groupby("group")["split"].nunique().max() > 1:
        raise ManifestError("a group occurs in both train and test")
    try:
        k = int(meta["inner_folds"])
    except (KeyError, TypeError, ValueError):
        raise ManifestError("metadata has no integer 'inner_folds'") from None
    test_fold = m.loc[m["split"] == "test", "inner_fold"]
    train_fold = m.loc[m["split"] == "train", "inner_fold"]
    if (test_fold != -1).any() or (~train_fold.isin(range(k))).any():
        raise ManifestError("inner_fold must be -1 for test and 0..inner_folds-1 for train")
    if m.groupby("group")["inner_fold"].nunique().max() > 1:
        raise ManifestError("a group occurs in two inner folds")
    corpus = corpus_sha256(df, cols)
    checks = {
        "algorithm": ALGORITHM,
        "splits_version": SPLITS_VERSION,
        "row_order": ROW_ORDER_POLICY,
        "textprep_version": TEXTPREP_VERSION,
        "taxonomy_sha256": tax.sha256,
        "corpus_sha256": corpus,
        "manifest_sha256": manifest_sha256(manifest),
        "grouping": grouping_stats(texts, min_prefix),
        "counts": _counts(manifest.assign(row_id=m_ids), k),
    }
    for name, value in checks.items():
        if meta.get(name) != value:
            raise ManifestError(f"metadata {name!r} does not match ({meta.get(name)!r} != {value!r})")
    try:
        recorded_input = input_sha256(corpus, meta)
    except KeyError as exc:
        raise ManifestError(f"metadata lacks {exc}") from None
    if meta.get("input_sha256") != recorded_input:
        raise ManifestError("metadata 'input_sha256' does not match the corpus and configuration")
    try:
        rebuilt = build_split(
            df, meta["seed"], meta["test_fraction"], k, min_prefix=min_prefix,
            max_group_rows=meta["max_group_rows"], columns=cols, taxonomy=tax,
        ).set_index("row_id")
    except KeyError as exc:
        raise ManifestError(f"metadata lacks {exc}") from None
    except ValueError as exc:
        raise ManifestError(f"rebuilding the split with the recorded parameters failed: {exc}") from None
    if not rebuilt[MANIFEST_COLUMNS[1:]].equals(m[MANIFEST_COLUMNS[1:]]):
        raise ManifestError("rebuilding the split with the recorded parameters gives a different manifest")


def assign_split(df: pd.DataFrame, manifest: pd.DataFrame, id_col: Optional[str] = None) -> pd.DataFrame:
    """Return ``df`` with ``group``, ``split`` and ``inner_fold`` columns from ``manifest``
    (joined on the id column recorded in the metadata, or ``id_col``)."""
    if id_col is None:
        id_col = {**DEFAULT_COLUMNS, **_metadata(manifest, None).get("columns", {})}["id"]
    ids = _row_ids(df, id_col)
    m = manifest.assign(row_id=manifest["row_id"].astype(str)).set_index("row_id")
    missing = set(ids) - set(m.index)
    if missing:
        raise ManifestError(f"{len(missing)} rows are not in the manifest")
    part = m.loc[ids, ["group", "split", "inner_fold"]]
    return df.assign(**{c: part[c].to_numpy() for c in part.columns})


ALL_ROWS = "(all)"


def balance_report(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    labels: Optional[Sequence[Sequence[str]]] = None,
    *,
    taxonomy: Optional[tx.Taxonomy] = None,
) -> pd.DataFrame:
    """Per-label support by split and inner fold, at row and at group level.

    ``labels`` (one label list per row of ``df``) defaults to ``stratification_labels``;
    pass e.g. one-step labels to check another target. A group counts for a label at group
    level if any of its rows carries it. Returns one row per (``level`` in ``rows`` /
    ``groups``, ``label``; ``label == "(all)"`` counts everything) with columns ``total``,
    ``train``, ``test``, ``fold_0``..``fold_{k-1}``, ``test_share`` (test / total),
    ``test_dev`` (``test_share`` - test fraction), ``max_fold_dev`` (largest
    ``|fold / train - 1 / k|``) and ``min_fold``. Sort by ``test_dev`` or ``max_fold_dev``
    to find the worst-balanced labels."""
    meta = _metadata(manifest, None)
    cols = {**DEFAULT_COLUMNS, **meta.get("columns", {})}
    k = int(meta["inner_folds"])
    frac = float(meta["test_fraction"])
    joined = assign_split(df, manifest, cols["id"])
    if labels is None:
        wls = [tx.watson_list(w) for w in df[cols["watson"]]]
        labels = stratification_labels(df[cols["relevant"]], wls, taxonomy)
    if len(labels) != len(df):
        raise ValueError("labels must have one entry per row of df")
    long = pd.DataFrame(
        [(g, s, f, lab) for g, s, f, row in zip(joined["group"], joined["split"], joined["inner_fold"], labels)
         for lab in [ALL_ROWS, *row]],
        columns=["group", "split", "inner_fold", "label"],
    )
    parts = ["train", "test"] + [f"fold_{f}" for f in range(k)]
    out = []
    for level, frame in (("rows", long), ("groups", long.drop_duplicates(["group", "label"]))):
        table = pd.DataFrame(0, index=sorted(long["label"].unique()), columns=["total", *parts])
        table["total"] = frame.groupby("label").size()
        for split in ("train", "test"):
            table[split] = frame[frame["split"] == split].groupby("label").size()
        for f in range(k):
            table[f"fold_{f}"] = frame[(frame["split"] == "train") & (frame["inner_fold"] == f)].groupby("label").size()
        table = table.fillna(0).astype(int)
        table.insert(0, "level", level)
        out.append(table)
    report = pd.concat(out).rename_axis("label").reset_index()
    fold_cols = [f"fold_{f}" for f in range(k)]
    report["test_share"] = report["test"] / report["total"]
    report["test_dev"] = report["test_share"] - frac
    train = report["train"].where(report["train"] > 0)
    report["max_fold_dev"] = report[fold_cols].div(train, axis=0).sub(1 / k).abs().max(axis=1)
    report["min_fold"] = report[fold_cols].min(axis=1)
    return report[["level", "label", "total", *parts, "test_share", "test_dev", "max_fold_dev", "min_fold"]]


def frequent_labelsets(
    labels: Iterable[Sequence[str]], min_count: int = 20, groups: Optional[Iterable[Any]] = None
) -> Union[Set[Tuple[str, ...]], Dict[Tuple[str, ...], Dict[str, int]]]:
    """Label sets (sorted tuples) occurring in at least ``min_count`` rows: the thesis rule
    that MLC label sets with fewer than 20 samples are discarded.

    Compute it on the rows of each FITTING partition only, i.e. the inner-train folds for
    model selection and the full train split for the final refit, and apply the result to
    the matching evaluation rows (inner validation fold, test). Computing it on the full
    corpus, or on train for an inner-fold fit, leaks evaluation label-set frequencies into
    the data selection.

    Without ``groups`` returns the set of frequent label sets. With ``groups`` (one group id
    per row, e.g. the manifest ``group`` column) returns ``{labelset: {"rows": n,
    "groups": g}}`` for the frequent label sets (threshold still on rows), so a label set
    carried by a few duplicated posts can be spotted."""
    rows = [tuple(sorted(row)) for row in labels]
    counts = Counter(rows)
    frequent = {labelset for labelset, n in counts.items() if n >= min_count}
    if groups is None:
        return frequent
    groups = list(groups)
    if len(groups) != len(rows):
        raise ValueError("groups must have one entry per label row")
    seen: Dict[Tuple[str, ...], Set[Any]] = {}
    for labelset, group in zip(rows, groups):
        if labelset in frequent:
            seen.setdefault(labelset, set()).add(group)
    return {ls: {"rows": counts[ls], "groups": len(seen[ls])} for ls in sorted(frequent)}
