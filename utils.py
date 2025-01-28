from collections import Counter
from copy import copy
from itertools import combinations
from typing import Optional, Union, Any, Dict, Tuple, List

import numpy
import numpy as np
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, \
    recall_score, hamming_loss, coverage_error, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

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


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 classifier: str,
                 hyperparameters: dict = None,
                 to_optimize: bool = False,
                 classes: list[str] = None,
                 random_state: int = 42):
        """
        Initializes the BaseClassifier.

        Parameters
        ----------
            classifier
                The name of the classifier.
            hyperparameters
                The dictionary of hyperparameters.
            to_optimize
                If the hyperparameters should be optimized.
            classes
                The list of classes.
            random_state
                The random state.
        """
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.to_optimize = to_optimize
        self.classes = classes
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = BASE_CLASSIFIERS[self.classifier](**self.hyperparameters)
        self.model.fit(X, y)
        return self

    def optimize(self, X_train, y_train, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X_train
            Data for training.
        y_train
            Target for training.
        X_val
            Data for validation.
        y_val
            Target for validation.

        Returns
        -------
        The optimized model.
        """
        if self.to_optimize:
            pbounds = CLASSIFIERS_BOUNDS[self.classifier]
            classifier_cls = BASE_CLASSIFIERS[self.classifier]

            self.model = BayesianOptimization(
                f=lambda **hp: f1_score(
                    y_true=y_val,
                    y_pred=classifier_cls(
                        **{k: (
                            PENALTY_DICT[round(v)] if k == 'penalty'
                            else SOLVER_DICT[round(v)] if k == 'solver'
                            else v
                        ) for k, v in hp.items()},
                    ).fit(X=X_train, y=y_train).predict(X=X_val)
                ),
                pbounds=pbounds,
                random_state=self.random_state
            )
            self.model.maximize(init_points=1, n_iter=5)
            self.model = classifier_cls(
                **{k: (
                    PENALTY_DICT[round(v)] if k == 'penalty'
                    else SOLVER_DICT[round(v)] if k == 'solver'
                    else v
                ) for k, v in self.model.max['params'].items()},
            ).fit(X=X_train, y=y_train)
        else:
            self.fit(X=X_train, y=y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true=y, y_pred=y_pred)
        print(f'Accuracy:\t{accuracy:.2f}')
        auc = roc_auc_score(y_true=y, y_score=y_pred, average='weighted')
        print(f'AUC:\t{auc:.2f}')
        report = classification_report(
            y_true=y,
            y_pred=y_pred,
            target_names=self.classes,
            zero_division=0
        )
        print(f'Classification report:\n{report}')

        metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "classification_report": report,
        }

        return metrics


class MultiLabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 classifier: str,
                 classes: Optional[list[str]],
                 hyperparameters: Optional[Union[list[dict], dict]] = None,
                 to_optimize: bool = False,
                 random_state: int = 42):
        """
        Initializes the MultiLabelClassifier.

        Parameters
        ----------
            classifier
                Name of the classifier.
            hyperparameters
                If it is a list of dictionaries, the classifiers will have the corresponding hyperparameters.
                If it is a dictionary, all the classifiers will have the same hyperparameters.
                If it is None, all the classifiers will have the default hyperparameters.
            classes
                Names of classes.
            random_state
                Random seed for reproducibility.
        """
        self.classifier = classifier
        self.classes = classes
        self.to_optimize = to_optimize
        self.random_state = random_state
        self.models = []  # Stores individual binary classifiers

        if isinstance(hyperparameters, dict) and not to_optimize:
            self.hyperparameters = [hyperparameters] * len(classes)
        elif isinstance(hyperparameters, list) and not to_optimize:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = [{}] * len(classes)

    def fit(self, X, y):
        """
        Trains the classifier for each label using Logistic Regression with SMOTE.

        Parameters:
        - X: array-like, shape (n_samples, n_features), training feature matrix.
        - y: array-like, shape (n_samples, n_labels), training target matrix.

        Returns:
        - self: object, the trained classifier.
        """
        for i in range(y.shape[1]):
            smote = SMOTE(random_state=self.random_state)

            # Apply SMOTE for current label
            X_resampled, y_resampled = smote.fit_resample(X=X, y=y[:, i])
            model = BaseClassifier(
                classifier=self.classifier,
                hyperparameters=self.hyperparameters[i],
                classes=self.classes,
                random_state=self.random_state
            )
            model.fit(X_resampled, y_resampled)
            self.models.append(model)
        return self

    def optimize(self, X_train, y_train, X_val=None, y_val=None):
        if self.to_optimize:
            for i in range(y_train.shape[1]):
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X=X_train, y=y_train[:, i])
                model = BaseClassifier(
                    classifier=self.classifier,
                    to_optimize=True,
                    classes=self.classes,
                    random_state=self.random_state
                )
                model.optimize(
                    X_train=X_resampled,
                    y_train=y_resampled,
                    X_val=X_val,
                    y_val=y_val[:, i]
                )
                self.models.append(model)
        else:
            self.fit(X=X_train, y=y_train)
        return self

    def predict(self, X):
        """
        Predicts labels for the given samples.

        Parameters
        ----------
            X
                shape (n_samples, n_features), input feature matrix.

        Returns
        -------
            shape (n_samples, n_labels), predicted label matrix.
        """
        n_labels = len(self.models)
        y_pred = np.zeros((X.shape[0], n_labels), dtype=int)

        for i, model in enumerate(self.models):
            y_pred[:, i] = model.predict(X)

        return y_pred

    def evaluate(self, X, y):
        """
        Evaluates the model on the given data.

        Parameters:
        - X: array-like, shape (n_samples, n_features), input feature matrix.
        - y: array-like, shape (n_samples, n_labels), true labels.

        Returns:
        - metrics: dict, evaluation metrics including accuracy, AUC, and classification report.
        """
        y_pred = self.predict(X)

        accuracy = accuracy_score(y_true=y, y_pred=y_pred)
        print(f'Accuracy:\t{accuracy:.2f}')
        auc = roc_auc_score(y_true=y, y_score=y_pred, average='weighted')
        print(f'AUC:\t{auc:.2f}')
        report = classification_report(
            y_true=y,
            y_pred=y_pred,
            target_names=self.classes,
            zero_division=0
        )
        print(f'Classification report:\n{report}')

        metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "classification_report": report,
        }

        return metrics


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
        np.random.shuffle(x=self.label_order)

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

    for idx, lbl_set in enumerate(y):
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
                target_names=classes
            )

    return model_performance


def display_assessments(evaluation: dict):
    for k in evaluation.keys():
        print(f'{k}:')
        print(f"Best classifier: {evaluation[k]['Classifier']}")
        print(f"Accuracy:\t{evaluation[k]['Accuracy']:.2f}")
        print(evaluation[k]['Classification'])
