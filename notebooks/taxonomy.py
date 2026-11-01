"""Thesis label taxonomy: Watson categories -> thesis labels per level.

The hierarchy and the Watson node-name mapping live in ``taxonomy.toml`` (next to this
file); this module only loads, validates and applies it. Every notebook and the System
must derive labels through these functions.

Levels: ``"top"`` (computer security, hardware, networking, operating systems, other,
software) and one sub level per top-level parent that has subcategories (``parents()``).

Label semantics (all functions return sorted lists):

* ``top_labels(wl, relevant, "two_step")``: the categorizer of the Two-Step System is
  trained on relevant posts only; irrelevant input raises ``ValueError``. Labels are the
  mapped top categories, ``["other"]`` iff none maps.
* ``top_labels(wl, relevant, "one_step")``: irrelevant posts are forced to ``["other"]``
  regardless of Watson; relevant posts as in ``two_step``. One-step ``other`` therefore
  covers BOTH irrelevant posts and relevant posts without a mapped top category (2,255
  relevant posts in the CyberPosts corpus, the figure in Methodology.tex:96). Never derive
  relevance from a top-level ``other`` alone: use the ``relevant`` column (or the filter).
* ``sub_labels(wl, parent)``: mapped subcategories of ``parent``, ``["other"]`` iff none.
  The sub-level ``other`` is internal ("none of the subcategories"); the System drops it
  and the post "remains in the high-level category". Which rows a sub-level module sees
  (e.g. rows whose top labels contain ``parent``) is the caller's decision.
"""

import ast
import hashlib
import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.toml")
TOP = "top"
SCHEMES = ("two_step", "one_step")


@dataclass(frozen=True)
class Taxonomy:
    other: str
    top_classes: Tuple[str, ...]
    top_map: Mapping[str, str]
    sub_classes: Mapping[str, Tuple[str, ...]]
    sub_maps: Mapping[str, Mapping[str, str]]
    sha256: str

    def parents(self) -> List[str]:
        """Top-level labels that have a sub level, sorted."""
        return sorted(self.sub_classes)

    def classes(self, level: str) -> List[str]:
        """Fixed class order of ``level`` (``"top"`` or a parent from ``parents()``)."""
        if level == TOP:
            return list(self.top_classes)
        if level in self.sub_classes:
            return list(self.sub_classes[level])
        raise ValueError(f"unknown level {level!r}; expected 'top' or one of {self.parents()}")

    def top_labels(self, watson_list: Iterable[str], relevant: Any, scheme: str) -> List[str]:
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
        relevant = _as_bool(relevant)
        if not relevant:
            if scheme == "two_step":
                raise ValueError("two_step top labels are defined for relevant posts only")
            return [self.other]
        return _map(watson_list, self.top_map, self.other)

    def sub_labels(self, watson_list: Iterable[str], parent: str) -> List[str]:
        if parent not in self.sub_maps:
            raise ValueError(f"{parent!r} has no sub level; parents: {self.parents()}")
        return _map(watson_list, self.sub_maps[parent], self.other)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"relevant must be a bool, got {type(value).__name__}: {value!r}")


def _map(watson_list: Iterable[str], mapping: Mapping[str, str], other: str) -> List[str]:
    if isinstance(watson_list, str):
        raise TypeError("watson_list must be a list of node names, not a string")
    mapped = {mapping[node] for node in watson_list if node in mapping}
    return sorted(mapped) if mapped else [other]


def _validate(raw: Dict[str, Any]) -> None:
    other = raw["other"]
    top = raw["top"]
    classes = top["classes"]
    if classes != sorted(classes) or len(set(classes)) != len(classes) or other not in classes:
        raise ValueError("top.classes must be sorted, unique and contain `other`")
    targets = set(top["watson"].values())
    if not targets <= set(classes) - {other}:
        raise ValueError(f"top.watson maps to unknown labels: {sorted(targets - set(classes))}")
    for parent, sub in raw.get("sub", {}).items():
        sc = sub["classes"]
        if parent not in classes or parent == other:
            raise ValueError(f"sub level {parent!r} is not a top-level class")
        if sc != sorted(sc) or len(set(sc)) != len(sc) or other not in sc:
            raise ValueError(f"sub.{parent}.classes must be sorted, unique and contain `other`")
        st = set(sub["watson"].values())
        if not st <= set(sc) - {other}:
            raise ValueError(f"sub.{parent}.watson maps to unknown labels: {sorted(st - set(sc))}")


def load_taxonomy(path: Union[str, Path, None] = None) -> Taxonomy:
    """Load and validate a taxonomy TOML (default ``taxonomy.toml``). ``sha256`` is the
    digest of the parsed content (canonical JSON), so comment edits do not change it."""
    with open(path or TAXONOMY_PATH, "rb") as f:
        raw = tomllib.load(f)
    _validate(raw)
    subs = raw.get("sub", {})
    return Taxonomy(
        other=raw["other"],
        top_classes=tuple(raw["top"]["classes"]),
        top_map=dict(raw["top"]["watson"]),
        sub_classes={p: tuple(s["classes"]) for p, s in subs.items()},
        sub_maps={p: dict(s["watson"]) for p, s in subs.items()},
        sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
    )


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    return load_taxonomy()


def _tax(taxonomy: Optional[Taxonomy]) -> Taxonomy:
    return default_taxonomy() if taxonomy is None else taxonomy


# Watson parsing: same semantics as notebooks.utils construct_hierarchy and
# flatten_dict_keys (re-implemented here so this module does not import nltk and
# sentence-transformers).


def _construct_hierarchy(categories: List[Dict]) -> Dict:
    tree: Dict = {}
    for entry in categories:
        node = tree
        for segment in entry["label"].strip("/").split("/"):
            node = node.setdefault(segment, {})
    return tree


def _flatten_dict_keys(tree: Any, acc: Optional[List] = None) -> List:
    acc = [] if acc is None else acc
    if isinstance(tree, dict):
        for key, value in tree.items():
            acc = _flatten_dict_keys(value, acc + [key])
    else:
        acc = [tree]
    return acc


def watson_list(raw: Union[str, Mapping, None]) -> List[str]:
    """Node names of all Watson category paths, in depth-first order.

    ``raw`` is the corpus ``watson`` field: a Python-literal dict string (parsed with
    ``ast.literal_eval``) or an already parsed dict; its ``categories`` entries have a
    path-like ``label``. Missing categories give ``[]``.
    """
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    if not isinstance(raw, Mapping):
        raise TypeError(f"watson field must be a dict or its string form, got {type(raw).__name__}")
    return _flatten_dict_keys(_construct_hierarchy(raw.get("categories", [])))


def top_labels(
    watson_list: Iterable[str], relevant: Any, scheme: str, taxonomy: Optional[Taxonomy] = None
) -> List[str]:
    """Top-level labels of a post under ``scheme`` (``two_step`` or ``one_step``); see the
    module docstring."""
    return _tax(taxonomy).top_labels(watson_list, relevant, scheme)


def sub_labels(
    watson_list: Iterable[str], parent: str, taxonomy: Optional[Taxonomy] = None
) -> List[str]:
    """Sub-level labels under ``parent``; ``["other"]`` (internal) iff none maps."""
    return _tax(taxonomy).sub_labels(watson_list, parent)


def classes(level: str, taxonomy: Optional[Taxonomy] = None) -> List[str]:
    """Fixed class order of ``level`` (``"top"`` or a parent with a sub level)."""
    return _tax(taxonomy).classes(level)


def parents(taxonomy: Optional[Taxonomy] = None) -> List[str]:
    """Top-level labels that have a sub level."""
    return _tax(taxonomy).parents()


def taxonomy_sha256(taxonomy: Optional[Taxonomy] = None) -> str:
    return _tax(taxonomy).sha256
