from typing import Optional, Union

import numpy as np
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
