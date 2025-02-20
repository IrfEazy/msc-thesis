import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from tqdm.notebook import tqdm


class CCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, order=None):
        """
        Initialize the Classifier Chain.

        Parameters
        ----------
        base_estimator
            The base classifier to be used for each binary classification task.
            If None, LogisticRegression is used by default.
        order
            The order of labels in the chain. If None, the natural order is used.
        """
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression()
        self.order = order
        self.chain = []

    def fit(self, X: numpy.ndarray, Y: numpy.ndarray) -> object:
        """
        Fit the classifier chain on the training data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        Y : numpy.ndarray
            Label matrix of shape (n_samples, n_labels).
        """
        n_samples, n_labels = Y.shape
        self.order_ = self.order if self.order is not None else range(n_labels)
        X_extended = numpy.copy(X)

        for i in self.order_:
            clf = clone(self.base_estimator)
            clf.fit(X_extended, Y[:, i])
            self.chain.append(clf)
            # Augment feature space with the current label's predictions
            predictions = clf.predict(X_extended).reshape(-1, 1)
            X_extended = numpy.hstack((X_extended, predictions))

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Predict labels for the given data.

        Parameters
        ----------
        X: numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Y_pred: numpy.ndarray
            Predicted label matrix of shape (n_samples, n_labels).
        """
        n_samples = X.shape[0]
        X_extended = numpy.copy(X)
        Y_pred = numpy.zeros((n_samples, len(self.chain)))

        for i, clf in enumerate(tqdm(self.chain, desc="Predicting for each classifier")):
            Y_pred[:, i] = clf.predict(X_extended)
            X_extended = numpy.hstack((X_extended, Y_pred[:, i].reshape(-1, 1)))

        return Y_pred

    def predict_proba(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Predict label probabilities for the given data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Y_proba : numpy.ndarray
            Predicted label probability matrix of shape (n_samples, n_labels).
        """
        n_samples = X.shape[0]
        X_extended = numpy.copy(X)
        Y_proba = numpy.zeros((n_samples, len(self.chain)))

        for i, clf in enumerate(tqdm(self.chain, desc="Predicting for each classifier")):
            if hasattr(clf, "predict_proba"):
                Y_proba[:, i] = clf.predict_proba(X_extended)[:, 1]
            else:
                Y_proba[:, i] = clf.decision_function(X_extended)
            X_extended = numpy.hstack((X_extended, Y_proba[:, i].reshape(-1, 1)))

        return Y_proba

    def evaluate(self, X: numpy.ndarray, Y_true: numpy.ndarray) -> dict[str, float]:
        """
        Evaluate the classifier chain on the given test data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        Y_true : numpy.ndarray
            True label matrix of shape (n_samples, n_labels).

        Returns
        -------
        metrics : dict
            Dictionary containing evaluation metrics.
        """
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average="micro")
        hamming = hamming_loss(Y_true, Y_pred)
        return {"accuracy": accuracy, "f1_micro": f1, "hamming_loss": hamming}
