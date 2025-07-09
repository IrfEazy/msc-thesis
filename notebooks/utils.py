import re
import string
import unicodedata

import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from nltk import WordNetLemmatizer, word_tokenize
from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    roc_auc_score,
)

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords, wordnet


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


def translate_src_categories(
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


def replace_txt_components(
    text: str,
    replace_emails: bool = True,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    replace_hashtags: bool = True,
    replace_phone_numbers: bool = True,
    custom_replacements: Optional[dict[str, str]] = None,
) -> str:
    """
    Replace specific text components (e.g., emails, URLs, mentions, hashtags) with placeholders.

    Parameters
    ----------
    text : str
        Input text to process.
    replace_emails : bool
        Whether to replace email addresses. Default is True.
    replace_urls : bool
        Whether to replace URLs. Default is True.
    replace_mentions : bool
        Whether to replace mentioned users. Default is True.
    replace_hashtags : bool
        Whether to replace hashtags. Default is True.
    replace_phone_numbers : bool
        Whether to replace phone numbers. Default is True.
    custom_replacements : dict
        Custom replacement rules as a dictionary. Default is None.

    Returns
    -------
    str : str
        Text with specified components replaced.
    """
    # Replace email addresses
    if replace_emails:
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

    # Replace URLs
    if replace_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Replace mentioned users
    if replace_mentions:
        text = re.sub(r"@\w+", "", text)

    # Replace hashtags
    if replace_hashtags:
        text = re.sub(r"#\w+", "", text)  # Remove hashtags entirely

    # Replace phone numbers
    if replace_phone_numbers:
        text = re.sub(
            r"\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b",
            "",
            text,
        )

    # Apply custom replacements if provided
    if custom_replacements is not None:
        for pattern, replacement in custom_replacements.items():
            text = re.sub(pattern, replacement, text)

    return text


def clean_text(
    text: str,
    remove_punctuation: bool = True,
    remove_emojis: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Clean and preprocess text data for machine learning tasks.

    Parameters
    ----------
    text : str
        Input text to be cleaned.
    remove_punctuation : bool
        Whether to remove punctuation. Default is True.
    remove_emojis : bool
        Whether to remove emojis and emoticons. Default is True.
    normalize_whitespace : bool
        Whether to normalize whitespace. Default is True.
    lowercase : bool
        Whether to convert text to lowercase. Default is True.

    Returns
    -------
    str : str
        Cleaned and preprocessed text.
    """
    # Convert text to lowercase if specified
    if lowercase:
        text = text.lower()

    # Remove punctuation if specified
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace if specified
    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    # Remove emojis and emoticons if specified
    if remove_emojis:
        # Remove emojis and emoticons using Unicode ranges
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
        # Remove additional emoticons and symbols
        text = re.sub(r"[\u2600-\u26FF\u2700-\u27BF]", "", text)

    # Normalize Unicode characters (e.g., convert accented characters to their base form)
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )

    return text


def get_wordnet_pos(treebank_tag):
    """
    Map Treebank POS tags to WordNet POS tags for lemmatization.

    Args:
        treebank_tag (str): Treebank POS tag.

    Returns:
        str: Corresponding WordNet POS tag.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no match


def lemmatize_text(text, lemmatizer=None):
    """
    Lemmatize text using WordNetLemmatizer with POS tagging for better accuracy.

    Parameters
    ----------
    text : str
        Input text to be lemmatized.
    lemmatizer : WordNetLemmatizer
        Lemmatizer instance.

    Returns
    -------
        str: Lemmatized text.
    """
    lemmatizer = WordNetLemmatizer() if lemmatizer is None else lemmatizer
    # Tokenize the text
    tokens = word_tokenize(text)

    # Get POS tags for each token
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize each token with its corresponding POS tag
    lemmatized_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)  # Convert Treebank tag to WordNet POS
        lemmatized_token = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmatized_tokens.append(lemmatized_token)

    # Join the lemmatized tokens into a single string
    return " ".join(lemmatized_tokens)


def remove_stopwords(text, language="english", custom_stopwords=None, lowercase=True):
    """
    Remove stopwords from the input text.

    Args:
        text (str): Input text to process.
        language (str): Language of the stopwords. Default is 'english'.
        custom_stopwords (set): Custom set of stopwords to use. Default is None.
        lowercase (bool): Whether to convert text to lowercase before processing. Default is True.

    Returns:
        str: Text with stopwords removed.
    """
    # Convert text to lowercase if specified
    text = text.lower() if lowercase else text

    # Tokenize the text
    tokens = word_tokenize(text)

    # Load stopwords
    stop_words = (
        set(stopwords.words(language))
        if custom_stopwords is None
        else set(custom_stopwords)
    )

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens into a single string
    return " ".join(filtered_tokens)


def load_word2vec_dict(
    model_path: Path, embedding_dim: int
) -> dict[Union[str, list[str]], ArrayLike]:
    embeddings_dict = {}
    with open(model_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[:-embedding_dim]
            word = " ".join(word) if type(word) is list else word
            vector = np.asarray([float(val) for val in values[-embedding_dim:]])
            embeddings_dict[word] = vector
    return embeddings_dict


def tokenizer_transform(
    x: pd.Series,
    embedder_addr: str,
    preprocessing_list: Optional[List[Callable[[str], str]]] = None,
) -> ArrayLike:
    """
    Generate embeddings for the sentences in the DataFrame.

    Parameters
    ----------
    x : pd.Series
        The DataFrame containing the sentences.
    embedder_addr : str
        Address of the embedder.
    preprocessing_list : Optional[list[Callable[[str], str]]]
        List of functions to apply to each sentence.

    Returns
    -------
    encoded_sentences : ArrayLike
        The embeddings of the sentences.
    """
    # Preprocess the text
    sentences = x.tolist()
    preprocessing_list = [] if preprocessing_list is None else preprocessing_list
    for preprocessor in preprocessing_list:
        sentences = [preprocessor(s) for s in sentences]
    model = SentenceTransformer(model_name_or_path=embedder_addr)
    return model.encode(sentences)


def preprocess_texts(list_str, model_path, embedding_dim):
    if embedding_dim is None:
        return tokenizer_transform(x=list_str, embedder_addr=model_path)

    word2vec_dict = load_word2vec_dict(model_path, embedding_dim)
    list_embedded_str = np.zeros((len(list_str), embedding_dim))

    for i, text in enumerate(list_str):
        tokens = re.findall(r"\w+|[{}]".format(re.escape(string.punctuation)), text)
        for token in tokens:
            try:
                list_embedded_str[i] += word2vec_dict[token.lower()]
            except KeyError:
                continue
    return list_embedded_str


def extract_models(classes, models, performance):
    attack_models = {}
    for idx in range(len(classes)):
        best_model = {}
        for model_name in models:
            temp_performance = models[model_name]["assess"]["report"][f"{idx}"][
                performance
            ]
            if len(best_model) == 0:
                best_model["name"] = model_name
                best_model["performance"] = temp_performance
                best_model["model"] = models[model_name]["model"]
            else:
                if best_model["performance"] < temp_performance:
                    best_model["name"] = model_name
                    best_model["performance"] = temp_performance
                    best_model["model"] = models[model_name]["model"]
        attack_models[f"{idx}"] = best_model
    return attack_models


def assess(Y: ArrayLike, Y_pred: ArrayLike) -> dict[str, float]:
    """
    Evaluate the model on the given data.

    Parameters
    ----------
    Y : ArrayLike of shape (n_samples, n_labels)
        The input features.
    Y_pred : ArrayLike of shape (n_samples, n_labels)
        The true binary label matrix.

    Returns
    -------
    metrics : dict[str, float]
        Dictionary containing accuracy, micro F1 score, and hamming loss.
    """
    accuracy = accuracy_score(Y, Y_pred)

    auc_score_micro = roc_auc_score(Y, Y_pred, average="micro")
    auc_score_macro = roc_auc_score(Y, Y_pred, average="macro")
    auc_score_weighted = roc_auc_score(Y, Y_pred, average="weighted")
    auc_score_samples = roc_auc_score(Y, Y_pred, average="samples")
    auc_per_label = roc_auc_score(Y, Y_pred, average=None)

    report = classification_report(Y, Y_pred, output_dict=True, zero_division=0.0)
    report["micro avg"]["auc"] = auc_score_micro
    report["macro avg"]["auc"] = auc_score_macro
    report["weighted avg"]["auc"] = auc_score_weighted
    report["samples avg"]["auc"] = auc_score_samples

    n_classes = Y.shape[1]
    class_names = [f"{i}" for i in range(n_classes)]
    for i, target in enumerate(class_names):
        if target in report:
            report[target]["auc"] = auc_per_label[i]
        else:
            # In case labels are not printed per class, you can store them separately
            report[target] = {"auc": auc_per_label[i]}

    hamming = hamming_loss(Y, Y_pred)
    return {"accuracy": accuracy, "hamming_loss": hamming, "report": report}
