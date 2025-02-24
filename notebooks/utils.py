from typing import Any, Dict, Iterable, List, Optional, Union


def construct_hierarchy(category_hierarchy: List[Dict]) -> Dict:
    """
    Build a tree-like structure (nested dictionary) from category labels.

    Parameters
    ----------
    category_hierarchy : list
        A list of categories, where each category has a 'label' key that contains a path-like string.

    Returns
    -------
    category_tree : dict
        A nested dictionary representing the tree structure.
    """
    category_tree = {}
    for category_entry in category_hierarchy:
        current_node = category_tree
        for category_segment in category_entry["label"].strip("/").split("/"):
            current_node = current_node.setdefault(category_segment, {})
    return category_tree


def flatten_dict_keys(
    input_dict: Union[Dict, Any], accumulated_keys: Optional[Iterable] = None
) -> List:
    """
    Recursively extract keys from a dictionary, building paths as a list.

    Parameters
    ----------
    d : dict or any
        The dictionary to extract keys from.
    path : list, optional
        A list to accumulate the path, default is None.

    Returns
    -------
    path : list
        A list of paths representing keys in the dictionary.
    """
    accumulated_keys = [] if accumulated_keys is None else accumulated_keys
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            accumulated_keys = flatten_dict_keys(
                input_dict=value, accumulated_keys=accumulated_keys + [key]
            )
    else:
        accumulated_keys = [input_dict]
    return accumulated_keys


def translate_source_categories(
    source_categories: List[str], category_mapping: Dict[str, str]
) -> List[str]:
    """
    Translate source categories to target categories using a mapping.

    Parameters
    ----------
    source_categories : list
        A list of source categories.
    category_mapping : dict
        A dictionary mapping source categories to target categories.

    Returns
    -------
    mapped_targets : list
        A list of target categories.
    """
    targets = set(category_mapping.keys()) & set(source_categories)
    mapped_targets = {category_mapping[category] for category in targets}
    return list(mapped_targets) if mapped_targets else ["other"]
