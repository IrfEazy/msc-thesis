import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple, DefaultDict, Union, Optional, Iterable

import numpy as np


def extract_keys(d: Union[dict, object], path: Optional[Iterable] = None) -> list:
    """
    Recursively extract keys from a dictionary, building paths as a list.

    Parameters
    ----------
    d
        The dictionary to extract keys from.
    path
        A list to accumulate the path, default is None.

    Returns
    -------
    path
        A list of paths representing keys in the dictionary.
    """
    if path is None:
        path = []

    if isinstance(d, dict):
        for key, value in d.items():
            path = extract_keys(d=value, path=path + [key])
    else:
        path = [d]

    return path


def build_tree(categories: list[dict]) -> dict:
    """
    Build a tree-like structure (nested dictionary) from category labels.

    Parameters
    ----------
    categories
        A list of categories, where each category has a 'label' key that contains a path-like string.

    Returns
    -------
    tree
        A nested dictionary representing the tree structure.
    """
    tree = {}

    for category in categories:
        current = tree

        for part in category['label'].strip('/').split('/'):
            current = current.setdefault(part, {})

    return tree


def merge_trees_with_counts(tree1: dict, tree2: dict, visit_count: defaultdict[Any, int]) -> dict:
    """
    Merge two trees recursively and count the visits to each node.

    Parameters
    ----------
    tree1
        The first tree to be merged.
    tree2
        It is the second tree to be merged.
    visit_count
        It is a dictionary that tracks the visit count for each node.

    Returns
    -------
    tree1
        The merged tree after processing both input trees.
    """
    for key, value in tree2.items():
        if key not in tree1:
            tree1[key] = value
        elif isinstance(value, dict) and isinstance(tree1[key], dict):
            merge_trees_with_counts(
                tree1=tree1[key],
                tree2=value,
                visit_count=visit_count
            )

        # Count visits for the node
        visit_count[key] += 1
    return tree1


def merge_all_trees_with_counts(trees: list[dict]) -> Tuple[dict, DefaultDict[Any, int]]:
    """
    Merge all trees into one general tree and count the visits to each node.

    Parameters
    ----------
    trees
        A list of trees (dictionaries) to be merged.

    Returns
    -------
    general_tree, visit_count
        The merged tree with all nodes, and a dictionary mapping each node to its visit count.

    """
    visit_count = defaultdict(int)
    unique_trees = [json.loads(s=json.dumps(obj=tree, sort_keys=True)) for tree in trees]
    general_tree = {}

    for tree in unique_trees:
        general_tree = merge_trees_with_counts(
            tree1=general_tree,
            tree2=tree,
            visit_count=visit_count
        )

    return general_tree, visit_count


def load_word2vec_dict(model_path: Path, embedding_dim: int) -> dict[Union[str, list[str]], np.ndarray[Any, np.dtype]]:
    embeddings_dict = {}

    f = open(model_path, 'r', encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[:-embedding_dim]

        if type(word) is list:
            word = ' '.join(word)

        vector = np.asarray([float(val) for val in values[-embedding_dim:]])
        embeddings_dict[word] = vector

    f.close()

    return embeddings_dict


def preprocess_texts(list_str: list[str], model_path: Path, embedding_dim: int) -> np.ndarray[Any, np.dtype]:
    word2vec_dict = load_word2vec_dict(
        model_path=model_path,
        embedding_dim=embedding_dim
    )
    list_embedded_str = np.zeros(shape=(len(list_str), embedding_dim))

    for i, text in enumerate(list_str):
        tokens = re.findall(r'\w+|[{}]'.format(re.escape(pattern=string.punctuation)), text)
        for token in tokens:
            try:
                list_embedded_str[i] += word2vec_dict[token.lower()]
            except KeyError:
                continue

    return list_embedded_str
