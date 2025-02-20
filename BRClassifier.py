import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score


class BRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        """
        Initialize the Binary Relevance classifier.

        Parameters:
        base_estimator: scikit-learn binary classifier, default=LogisticRegression()
            The base classifier to use for each binary problem.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression()
        self.base_classifier = base_estimator

    def fit(self, X: numpy.ndarray, Y: numpy.ndarray):
        """
        Fit the Calibrated Label Ranking classifier.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The training input samples.
        Y : numpy.ndarray of shape (n_samples, n_labels)
            Binary indicator matrix with 1 indicating that the label is relevant.

        Returns
        -------
        self : object
        """
        n_labels = Y.shape[1]
        self.classifiers_ = []

        for i in range(n_labels):
            clf = clone(self.base_classifier)
            if hasattr(clf, "fit"):
                clf.fit(X, Y[:, i])
            else:
                raise Exception("Base classifier does not have a fit function")
            self.classifiers_.append(clf)

        return self

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Predict labels for the given data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input features.

        Returns
        -------
        Y_pred : numpy.ndarray of shape (n_samples, n_labels)
            The predicted binary label matrix.
        """
        n_samples = X.shape[0]
        n_labels = len(self.classifiers_)
        Y_pred = numpy.zeros((n_samples, n_labels))

        for i, clf in enumerate(self.classifiers_):
            if hasattr(clf, "predict"):
                Y_pred[:, i] = clf.predict(X)
            else:
                raise Exception("Classifiers do not have a predict function")

        return Y_pred

    def predict_proba(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Predict label probabilities for the given data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input features.

        Returns
        -------
        Y_proba : numpy.ndarray of shape (n_samples, n_labels)
            The predicted probability matrix.
        """
        n_samples = X.shape[0]
        n_labels = len(self.classifiers_)
        Y_proba = numpy.zeros((n_samples, n_labels))

        for i, clf in enumerate(self.classifiers_):
            if hasattr(clf, "predict_proba"):
                Y_proba[:, i] = clf.predict_proba(X)[:, 1]
            else:
                if hasattr(clf, "decision_function"):
                    Y_proba[:, i] = clf.decision_function(X)
                    # Apply sigmoid to convert to probabilities
                    Y_proba[:, i] = 1 / (1 + numpy.exp(-Y_proba[:, i]))
                else:
                    raise Exception("Classifiers have neither predict_proba and decision_function functions")

        return Y_proba

    def evaluate(self, X: numpy.ndarray, Y: numpy.ndarray) -> dict:
        """
        Evaluate the model on the given data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input features.
        Y : numpy.ndarray of shape (n_samples, n_labels)
            The true binary label matrix.

        Returns
        -------
        metrics : dict
            A dictionary containing evaluation metrics.
        """
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y, Y_pred)
        f1 = f1_score(Y, Y_pred, average="micro")
        hamming = hamming_loss(Y, Y_pred)
        return {"accuracy": accuracy, "f1_micro": f1, "hamming_loss": hamming}
