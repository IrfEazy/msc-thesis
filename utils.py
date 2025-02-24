import re
import unicodedata
from collections import Counter
from copy import copy
from itertools import combinations
from typing import Optional, Union, Any, Dict, Tuple, List, Callable

import nltk
import numpy
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, \
    recall_score, hamming_loss, coverage_error, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from tqdm.notebook import tqdm
from xgboost import XGBClassifier

from preprocess_functions import get_wordnet_pos

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')

BASE_CLASSIFIERS = {
    'logistic_regression': LogisticRegression,
    'gaussian_naive_bayes': GaussianNB,
    'decision_tree': DecisionTreeClassifier,
    'svm': SVC,
    'random_forest': RandomForestClassifier,
    'xgb': XGBClassifier
}

CLASSIFIERS_BOUNDS = {
    'logistic_regression': {
        'C': (1e-2, 1e2),
        'penalty': (0, 1),
        'solver': (0, 0.25)
    }
}

PENALTY_DICT = {
    0: 'l1',
    1: 'l2'
}

SOLVER_DICT = {
    0: 'liblinear'
}






class CalibratedLabelRankClassifier(BaseEstimator):
    def __init__(self, classifier: Any = None, classes: List[str] = None, random_state: int = None):
        self.classifier = classifier
        self.classes = classes
        self.random_state = random_state
        self.pairwise_classifiers = {}
        self.artificial_classifiers = {}

    def fit(self, x: np.ndarray[Any, np.dtype[float]], y: List[List[int]]):
        for (li, lj) in combinations(iterable=range(len(self.classes)), r=2):
            x_pair = []
            y_pair = []

            for idx, labels in enumerate(y):
                binary_labels = [1 if label in labels else 0 for label in self.classes]

                if binary_labels[li] != binary_labels[lj]:
                    x_pair.append(x[idx])
                    y_pair.append(1 if binary_labels[li] > binary_labels[lj] else 0)

            if x_pair:
                x_pair = np.array(x_pair)
                y_pair = np.array(y_pair)
                model = copy(self.classifier)

                self.pairwise_classifiers[(li, lj)] = model.fit(
                    X=x_pair,
                    y=y_pair,
                    sample_weight=compute_sample_weight(class_weight='balanced', y=y_pair)
                )

        for li in range(len(self.classes)):
            x_artificial = []
            y_artificial = []

            for idx, labels in enumerate(y):
                binary_labels = [1 if label in labels else 0 for label in self.classes]
                x_artificial.append(x[idx])
                y_artificial.append(binary_labels[li])

            x_artificial = np.array(x_artificial)
            y_artificial = np.array(y_artificial)
            model = copy(self.classifier)

            self.artificial_classifiers[li] = model.fit(
                X=x_artificial,
                y=y_artificial,
                sample_weight=compute_sample_weight(class_weight='balanced', y=y_artificial)
            )

        return self

    def predict(self, x):
        rankings = []
        bipartitions = []
        num_samples = x.shape[0]
        num_labels = len(self.classes)

        for instance in x:  # Iterate over each instance
            # 1. Predict pairwise preferences
            scores = np.zeros(shape=num_labels)  # Initialize scores for each label

            for (li, lj), model in self.pairwise_classifiers.items():
                y_proj = model.predict(X=instance.reshape(1, -1))[0]  # Predict for this pair

                if y_proj == 1:  # li is more relevant
                    scores[li] += 1
                else:  # lj is more relevant
                    scores[lj] += 1

            # 2. Rank labels based on scores
            label_ranking = np.argsort(a=-scores)  # Sort labels by descending scores
            rankings.append(label_ranking)

            # 3. Predict relevance using artificial classifiers
            relevance = []
            for li, model in self.artificial_classifiers.items():
                y_proj = model.predict(X=instance.reshape(1, -1))[0]
                relevance.append(y_proj)

            # Split ranking into relevant and irrelevant labels
            relevant = [label for label in label_ranking if relevance[label] == 1]
            irrelevant = [label for label in label_ranking if relevance[label] == 0]
            bipartitions.append((relevant, irrelevant))

        y_proj = np.zeros(shape=(num_samples, num_labels), dtype=int)

        for i, (ranking, bipartition) in enumerate(zip(rankings, bipartitions)):
            relevant_labels, _ = bipartition  # Extract relevant labels

            for label in relevant_labels:
                y_proj[i, label] = 1  # Mark relevant labels as 1

        return y_proj


class ChainOfClassifiers(BaseEstimator):
    def __init__(self, classifier: Any = None, classes: List[str] = None, random_state: int = None):
        self.classifier = classifier
        self.classes = classes
        self.random_state = random_state
        self.classifiers_chain = []
        self.label_order = None

    def fit(self, x, y):
        self.label_order = np.arange(len(self.classes))
        x_augmented = x.copy()

        # Randomize label order
        np.random.shuffle(self.label_order)

        # Sequentially train classifiers in the chain
        for _, label_idx in enumerate(self.label_order):
            # Train the classifier for the current label
            model = copy(self.classifier)

            model.fit(
                X=x_augmented,
                y=y[:, label_idx],
                sample_weight=compute_sample_weight(class_weight='balanced', y=y[:, label_idx])
            )

            self.classifiers_chain.append(model)

            # Predict current label for all instances to augment features
            model_prediction = model.predict(X=x_augmented).reshape(-1, 1)

            # Augment feature space with predictions from previous classifiers
            x_augmented = np.hstack(tup=(x_augmented, model_prediction))

        return self

    def predict(self, x):
        num_samples = x.shape[0]
        num_labels = len(self.classes)
        y = np.zeros(shape=(num_samples, num_labels), dtype=int)

        x_augmented = x.copy()  # Start with original features

        for i, model in enumerate(self.classifiers_chain):
            # Predict current label
            yi = model.predict(X=x_augmented).reshape(-1, 1)

            # Augment features for the next classifier
            x_augmented = np.hstack((x_augmented, yi))

            # Store prediction in the correct position in y
            y[:, self.label_order[i]] = yi.ravel()

        return y


class LabelPowersetClassifier(BaseEstimator):
    def __init__(self, classifier: Any, label_map: Dict[int, Tuple[Any]], random_state: int = None):
        self.classifier = classifier
        self.label_map = label_map
        self.random_state = random_state

    def fit(self, x, y):
        model = copy(self.classifier)
        self.classifier = model.fit(X=x, y=y)
        return self

    def predict(self, x):
        return np.array([list(self.label_map[yp]) for yp in self.classifier.predict(x)])


class PrunedSetsClassifier(BaseEstimator):
    def __init__(self, classifier: Any, label_map: dict[Any, Any], max_sub_samples: int = 3, pruning_threshold: int = 5,
                 random_state: int = None):
        self.classifier = classifier
        self.label_map = label_map
        self.max_sub_samples = max_sub_samples
        self.pruning_threshold = pruning_threshold
        self.random_state = random_state

    def fit(self, x, y):
        model = copy(self.classifier)
        self.classifier = model.fit(X=x, y=y)
        return self

    def predict(self, x):
        return np.array([list(self.label_map[yp]) for yp in self.classifier.predict(x)])


class ConditionalDependencyNetwork(BaseEstimator):
    def __init__(self, classifier, num_iterations=100, burn_in=10):
        self.classifier = classifier
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.models = []
        self.num_labels = None

    def fit(self, x, y):
        """
        Train the Conditional Dependency Network.

        Parameters:
        - X: Feature matrix (n_samples x n_features)
        - Y: Label matrix (n_samples x n_labels)
        """
        self.num_labels = y.shape[1]
        self.models = []

        for label_idx in range(self.num_labels):
            # Create dataset for current label
            x_augmented = np.hstack((x, np.delete(y, label_idx, axis=1)))  # Remove current label
            y_label = y[:, label_idx]

            # Train a binary classifier
            model = copy(self.classifier)
            model.fit(x_augmented, y_label)
            self.models.append(model)

        return self

    def _sample_label(self, x, y_current, label_idx):
        """
        Sample a value for a single label using its conditional probability model.

        Parameters:
        - X: Feature matrix
        - Y_current: Current label matrix (with possibly updated values)
        - label_idx: Index of the label to sample

        Returns:
        - Updated value for the label
        """
        x_augmented = np.hstack((x, np.delete(y_current, label_idx, axis=1)))
        probabilities = self.models[label_idx].predict_proba(x_augmented)[:, 1]
        return (probabilities > np.random.rand(len(probabilities))).astype(int)

    def predict(self, x):
        """
        Predict labels for new samples using Gibbs sampling.

        Parameters:
        - X: Feature matrix (n_samples x n_features)

        Returns:
        - Y_pred: Predicted label matrix (n_samples x n_labels)
        """
        # Initialize labels randomly
        y_current = np.random.randint(0, 2, size=(x.shape[0], self.num_labels))
        y_stationary = np.zeros_like(a=y_current)

        for iteration in range(self.num_iterations):
            for label_idx in range(self.num_labels):
                # Resample each label
                y_current[:, label_idx] = self._sample_label(x, y_current, label_idx)

            # Optional: Discard initial samples (burn-in)
            if iteration >= self.burn_in:
                y_stationary = y_current  # Save samples after burn-in for averaging or post-processing

        return y_stationary


class MetaBinaryRelevance(BaseEstimator):
    def __init__(self, classifier, use_cross_val=False, n_splits=5):
        """
        Parameters
        ----------
        classifier
            Base binary classifier to use for each label.
        use_cross_val
            Whether to use cross-validation for generating Stage 1 predictions.
        n_splits
            Number of folds for cross-validation if ``use_cross_val`` is True.
        """
        self.classifier = classifier
        self.use_cross_val = use_cross_val
        self.n_splits = n_splits
        self.stage1_models = []
        self.stage2_models = []
        self.num_labels = None

    def fit(self, x, y):
        """
        Train the MBR model.

        Parameters
        ----------
        x
            Feature matrix (n_samples x n_features)
        y
            Label matrix (n_samples x n_labels)
        """
        self.num_labels = y.shape[1]
        stage1_predictions = np.zeros_like(y, dtype=float)

        # Stage 1: Train binary classifiers and generate predictions
        for label_idx in range(self.num_labels):
            model = copy(x=self.classifier)

            # Use cross-validation to generate out-of-fold predictions
            predictions = cross_val_predict(
                estimator=model,
                X=x,
                y=y[:, label_idx],
                cv=self.n_splits,
                method='predict_proba'
            )[:, 1]
            # Train on the full training set and generate predictions
            model.fit(X=x, y=y[:, label_idx])
            # predictions = model.predict_proba(X=x)[:, 1]

            self.stage1_models.append(model)
            stage1_predictions[:, label_idx] = predictions

        # Augment the feature set with Stage 1 predictions
        x_augmented = np.hstack(tup=(x, stage1_predictions))

        # Stage 2: Train another set of binary classifiers
        for label_idx in range(self.num_labels):
            model = copy(self.classifier)
            model.fit(x_augmented, y[:, label_idx])
            self.stage2_models.append(model)

        return self

    def predict(self, x):
        """
        Predict labels for new samples.

        Parameters
        ----------
        x
            Feature matrix (n_samples x n_features)

        Returns
        -------
        y_pred
            Predicted label matrix (n_samples x n_labels)
        """
        # Generate predictions from Stage 1 models
        stage1_predictions = np.zeros(shape=(x.shape[0], self.num_labels), dtype=float)

        for label_idx, model in enumerate(self.stage1_models):
            stage1_predictions[:, label_idx] = model.predict_proba(x)[:, 1]

        # Augment the feature set with Stage 1 predictions
        x_augmented = np.hstack(tup=(x, stage1_predictions))

        # Generate predictions from Stage 2 models
        y_pred = np.zeros(shape=(x.shape[0], self.num_labels), dtype=int)

        for label_idx, model in enumerate(self.stage2_models):
            y_pred[:, label_idx] = model.predict(x_augmented)

        return y_pred


def prune_and_subsample(
        x: list[Any],
        y: list[list[Any]],
        pruning_threshold: float,
        max_sub_samples: int
) -> tuple[
    numpy.ndarray[Any, numpy.dtype],
    numpy.ndarray[Any, numpy.dtype],
    dict[int, Union[tuple[Any, ...], tuple[int, ...]]],
    numpy.ndarray[Any, numpy.dtype]
]:
    """
    Prune and subsample the given samples.

    Parameters
    ----------
    x
    y
    pruning_threshold
    max_sub_samples

    Returns
    -------

    """
    # Count occurrences of each label set
    label_counts = Counter(tuple(lbl_set) for lbl_set in y)
    frequent_label_sets = {k for k, v in label_counts.items() if v >= pruning_threshold}
    x_new = []
    y_new = []
    index_mapping = []
    label_map = {}
    class_idx = 0

    for idx, lbl_set in enumerate(tqdm(y)):
        lbl_tuple = tuple(lbl_set)

        if lbl_tuple in frequent_label_sets:
            # Preserve frequent label sets
            if lbl_tuple not in label_map:
                label_map[lbl_tuple] = class_idx
                class_idx += 1

            x_new.append(x[idx])
            y_new.append(label_map[lbl_tuple])
            index_mapping.append(idx)
        else:
            # Subsample infrequent label sets
            lbl_subsets = []

            for i in range(2 ** len(lbl_tuple)):
                lbl_subset = tuple((lbl_tuple[j] if (i >> j) & 1 else 0) for j in range(len(lbl_tuple)))
                lbl_subsets.append(lbl_subset)

            lbl_subsets = [subset for subset in lbl_subsets if subset in frequent_label_sets]
            lbl_subsets = lbl_subsets[:max_sub_samples]

            for subset in lbl_subsets:
                if subset not in label_map:
                    label_map[subset] = class_idx
                    class_idx += 1

                x_new.append(x[idx])
                y_new.append(label_map[subset])
                index_mapping.append(idx)

    return np.array(x_new), np.array(y_new), {v: k for k, v in label_map.items()}, np.array(index_mapping)


def assess_models(
        x: np.ndarray[Any, np.dtype[float]],
        y: np.ndarray[Any, np.dtype[int]],
        technique: dict[str, Any],
        classes: List[str]
) -> dict[str, Any]:
    model_performance = {
        'Accuracy': 0
    }

    for k, v in technique.items():
        y_predict = v.predict(x)
        accuracy_i = accuracy_score(y_true=y, y_pred=y_predict)

        if model_performance['Accuracy'] < accuracy_i:
            model_performance['Classifier'] = k
            model_performance['Model'] = v
            model_performance['Accuracy'] = accuracy_i

            model_performance['Precision example-based'] = precision_score(
                y_true=y,
                y_pred=y_predict,
                average='samples',
                zero_division=0
            )

            model_performance['Recall example-based'] = recall_score(
                y_true=y,
                y_pred=y_predict,
                average='samples',
                zero_division=0
            )

            model_performance['F1 example-based'] = f1_score(
                y_true=y,
                y_pred=y_predict,
                average='samples',
                zero_division=0
            )

            model_performance['Hamming loss'] = hamming_loss(y_true=y, y_pred=y_predict)

            model_performance['Micro precision'] = precision_score(
                y_true=y,
                y_pred=y_predict,
                average='micro',
                zero_division=0
            )

            model_performance['Micro recall'] = recall_score(
                y_true=y,
                y_pred=y_predict,
                average='micro',
                zero_division=0
            )

            model_performance['Micro F1'] = f1_score(
                y_true=y,
                y_pred=y_predict,
                average='micro',
                zero_division=0
            )

            model_performance['Macro precision'] = precision_score(
                y_true=y,
                y_pred=y_predict,
                average='macro',
                zero_division=0
            )

            model_performance['Macro recall'] = recall_score(
                y_true=y,
                y_pred=y_predict,
                average='macro',
                zero_division=0
            )

            model_performance['Macro F1'] = f1_score(
                y_true=y,
                y_pred=y_predict,
                average='macro',
                zero_division=0
            )

            model_performance['Coverage'] = coverage_error(y_true=y, y_score=y_predict)
            model_performance['Classification'] = classification_report(
                y_true=y,
                y_pred=y_predict,
                target_names=classes,
                zero_division=0
            )

    return model_performance


def display_assessments(evaluation: dict):
    for k in evaluation.keys():
        print(f'{k}:')
        print(f"Best classifier: {evaluation[k]['Classifier']}")
        print(f"Accuracy:\t{evaluation[k]['Accuracy']:.2f}")
        print(evaluation[k]['Classification'])


def sentence_embedding(sentence, tokenizer, model):
    """
    Generate a BERT embedding for a single sentence.

    Args:
        sentence (str): Input sentence.
        tokenizer: BERT tokenizer.
        model: BERT model.

    Returns:
        numpy.ndarray: Sentence embedding.
    """
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return np.array(outputs.last_hidden_state.mean(dim=1).detach().numpy()).squeeze()


def embed_sentences(sentences, tokenizer, model):
    """
    Generate BERT embeddings for a list of sentences.

    Args:
        sentences (list): List of sentences.
        tokenizer: BERT tokenizer.
        model: BERT model.

    Returns:
        list: List of sentence embeddings.
    """
    embeddings = []
    sentences = [replace_text_components(text=s) for s in sentences]
    sentences = [clean_text(text=s) for s in sentences]
    sentences = [lemmatize_text(text=s) for s in sentences]
    sentences = [remove_stopwords(text=s) for s in sentences]
    for sentence in sentences:
        embedding = sentence_embedding(sentence, tokenizer, model)
        embeddings.append(embedding)
    return embeddings


def embed_sentences_batch(sentences, tokenizer, model, batch_size=32):
    embeddings = []
    sentences = [replace_text_components(text=s) for s in sentences]
    sentences = [clean_text(text=s) for s in sentences]
    sentences = [lemmatize_text(text=s) for s in sentences]
    sentences = [remove_stopwords(text=s) for s in sentences]
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings


def replace_text_components(text, replace_emails=True, replace_urls=True, replace_mentions=True, replace_hashtags=True,
                            replace_phone_numbers=True, custom_replacements=None):
    """
    Replace specific text components (e.g., emails, URLs, mentions, hashtags) with placeholders.

    Args:
        text (str): Input text to process.
        replace_emails (bool): Whether to replace email addresses. Default is True.
        replace_urls (bool): Whether to replace URLs. Default is True.
        replace_mentions (bool): Whether to replace mentioned users. Default is True.
        replace_hashtags (bool): Whether to replace hashtags. Default is True.
        replace_phone_numbers (bool): Whether to replace phone numbers. Default is True.
        custom_replacements (dict): Custom replacement rules as a dictionary. Default is None.

    Returns:
        str: Text with specified components replaced.
    """
    # Replace email addresses
    if replace_emails:
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Replace URLs
    if replace_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Replace mentioned users
    if replace_mentions:
        text = re.sub(r'@\w+', '', text)

    # Replace hashtags
    if replace_hashtags:
        text = re.sub(r'#\w+', '', text)  # Remove hashtags entirely

    # Replace phone numbers
    if replace_phone_numbers:
        text = re.sub(r'\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b', '', text)

    # Apply custom replacements if provided
    if custom_replacements is not None:
        for pattern, replacement in custom_replacements.items():
            text = re.sub(pattern, replacement, text)

    return text


def clean_text(text, remove_punctuation=True, remove_emojis=True, normalize_whitespace=True, lowercase=True):
    """
    Clean and preprocess text data for machine learning tasks.

    Args:
        text (str): Input text to be cleaned.
        remove_punctuation (bool): Whether to remove punctuation. Default is True.
        remove_emojis (bool): Whether to remove emojis and emoticons. Default is True.
        normalize_whitespace (bool): Whether to normalize whitespace. Default is True.
        lowercase (bool): Whether to convert text to lowercase. Default is True.

    Returns:
        str: Cleaned and preprocessed text.
    """
    # Convert text to lowercase if specified
    if lowercase:
        text = text.lower()

    # Remove punctuation if specified
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace if specified
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    # Remove emojis and emoticons if specified
    if remove_emojis:
        # Remove emojis and emoticons using Unicode ranges
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # Remove additional emoticons and symbols
        text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)

    # Normalize Unicode characters (e.g., convert accented characters to their base form)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text


def lemmatize_text(text, lemmatizer=WordNetLemmatizer()):
    """
    Lemmatize text using WordNetLemmatizer with POS tagging for better accuracy.

    Args:
        text (str): Input text to be lemmatized.
        lemmatizer (WordNetLemmatizer): Lemmatizer instance. Default is WordNetLemmatizer().

    Returns:
        str: Lemmatized text.
    """
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
    return ' '.join(lemmatized_tokens)


def remove_stopwords(text, language='english', custom_stopwords=None, lowercase=True):
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
    if lowercase:
        text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Load stopwords
    if custom_stopwords is not None:
        stop_words = set(custom_stopwords)
    else:
        stop_words = set(stopwords.words(language))

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens into a single string
    return ' '.join(filtered_tokens)


def tokenizer_transform(
        x: pd.Series,
        embedder_addr: str,
        preprocessing_list: Optional[list[Callable[[str], str]]] = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Generate embeddings for the sentences in the DataFrame.

    Args:
        x (pd.Series): The DataFrame containing the sentences.
        embedder_addr (str): Address of the embedder.
        preprocessing_list (list[callable]): List of functions to apply to each sentence.
    """
    # Preprocess the text
    sentences = x.tolist()

    if preprocessing_list is None:
        preprocessing_list = []

    for preprocessor in preprocessing_list:
        sentences = [preprocessor(s) for s in sentences]

    model = SentenceTransformer(model_name_or_path=embedder_addr)
    return model.encode(sentences)


def tokenizer_transform_old(
        x: pd.Series,
        tokenizer,
        embedder,
        text_preprocessors=None,
        batch_size: int = 128,
        max_length: int = 512
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Generate embeddings for the sentences in the DataFrame using a tokenizer and model.

    Args:
        x (pd.Series): The DataFrame containing the sentences.
        batch_size (int): The batch size for preprocessing sentences.
        max_length (int): The maximum length of the tokenized input.
    """
    # Preprocess the text
    sentences = x.tolist()

    if text_preprocessors is None:
        text_preprocessors = []

    for preprocessor in text_preprocessors:
        sentences = [preprocessor(s) for s in sentences]

    # Generate embeddings in batches
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_length
        )
        outputs = embedder(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.extend(batch_embeddings)

    # Add embeddings to the DataFrame
    return np.array(embeddings)
