import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.utils import check_X_y, check_array


class CLRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        """
        Parameters
        ----------
        base_estimator : scikit-learn classifier, default=LogisticRegression()
            The base binary classifier to be used for both pairwise and auxiliary classifiers.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression()
        self.base_estimator = base_estimator
        self.q_ = None
        self.pairwise_classifiers_ = None
        self.aux_classifiers_ = None

    def fit(self, X: numpy.ndarray, Y: numpy.ndarray) -> object:
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
        # Validate inputs
        X, Y = check_X_y(X, Y, multi_output=True, accept_sparse=True)
        Y = numpy.array(Y)
        n_samples, self.q_ = Y.shape

        # Initialize dictionaries for pairwise and auxiliary classifiers
        self.pairwise_classifiers_ = {}
        self.aux_classifiers_ = {}

        # Train pairwise classifiers for each pair (j, k), j < k
        for j in range(self.q_):
            for k in range(j + 1, self.q_):
                X_pair = []
                y_pair = []
                # For each instance, consider it only if labels differ
                for i in range(n_samples):
                    if Y[i, j] != Y[i, k]:
                        X_pair.append(X[i])
                        # Define target: 1 if label j is relevant, 0 if label k is relevant
                        y_pair.append(1 if Y[i, j] == 1 else 0)
                if len(X_pair) > 0:
                    X_pair = numpy.array(X_pair)
                    y_pair = numpy.array(y_pair)
                    clf = clone(self.base_estimator)
                    if hasattr(clf, "fit"):
                        clf.fit(X_pair, y_pair)
                    else:
                        raise Exception("Base estimator does not have a fit function")
                    self.pairwise_classifiers_[(j, k)] = clf
        # Train auxiliary classifiers (for each label vs. the virtual label)
        for j in range(self.q_):
            clf_aux = clone(self.base_estimator)
            if hasattr(clf_aux, "fit"):
                clf_aux.fit(X, Y[:, j])
            else:
                raise Exception("Base estimator does not have a fit function")
            self.aux_classifiers_[j] = clf_aux

        return self

    def predict_proba(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Compute calibrated scores for each label.

        For each instance, the pairwise classifiers produce a normalized vote count (between 0 and 1)
        for each label. In parallel, the auxiliary classifiers produce a probability of label relevance.
        The calibrated score is the difference: (normalized vote) - (auxiliary probability).

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        calibrated_scores : numpy.ndarray of shape (n_samples, n_labels)
            Calibrated scores for each label. A positive score suggests relevance.
        """
        X = check_array(X, accept_sparse=True)
        if hasattr(X, "shape"):
            n_samples = X.shape[0]
        else:
            raise Exception("Input samples do not have shape function")
        votes = numpy.zeros((n_samples, self.q_))

        # Aggregate votes from pairwise classifiers
        for (j, k), clf in self.pairwise_classifiers_.items():
            if hasattr(clf, "predict"):
                preds = clf.predict(X)
            else:
                raise Exception("Classifiers do not have a predict function")
            # For each sample, if prediction is 1 then label j wins; otherwise label k wins.
            for i, pred in enumerate(preds):
                if pred == 1:
                    votes[i, j] += 1
                else:
                    votes[i, k] += 1

        # Normalize votes: maximum votes per label is (q - 1)
        norm_votes = votes / (self.q_ - 1)

        # Obtain auxiliary probabilities for each label
        aux_probs = numpy.zeros((n_samples, self.q_))
        for j, clf in self.aux_classifiers_.items():
            # Assuming the estimator has a predict_proba method;
            # otherwise one might use decision_function and calibrate it.
            if hasattr(clf, "predict_proba"):
                aux_probs[:, j] = clf.predict_proba(X)[:, 1]
            else:
                if hasattr(clf, "decision_function"):
                    aux_probs[:, j] = clf.decision_function(X)
                    # Apply sigmoid to convert to probabilities
                    aux_probs[:, j] = 1 / (1 + numpy.exp(-aux_probs[:, j]))
                else:
                    raise Exception("Classifiers have neither predict_proba and decision_function functions")

        # Calibrated score: if norm_votes exceed the auxiliary probability, label is deemed relevant.
        calibrated_scores = norm_votes - aux_probs
        return calibrated_scores

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Predict the multi-label set for each instance.

        A label is predicted as relevant if its calibrated score is positive.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples, n_labels)
            Binary indicator matrix of predicted labels.
        """
        calibrated_scores = self.predict_proba(X)
        predictions = (calibrated_scores > 0).astype(int)
        return predictions

    def evaluate(self, X: numpy.ndarray, Y_true: numpy.ndarray) -> dict:
        """
        Evaluate the classifier using standard multi-label metrics.

        The function computes:
          - Subset Accuracy: fraction of samples with exactly correct label set.
          - Micro-averaged F1 score.
          - Hamming Loss.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.
        Y_true : numpy.ndarray of shape (n_samples, n_labels)
            True binary indicator matrix of labels.

        Returns
        -------
        metrics : dict
            Dictionary containing accuracy, micro F1 score, and hamming loss.
        """
        Y_true = numpy.array(Y_true)
        Y_pred = self.predict(X)
        acc = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average="micro")
        hamming = hamming_loss(Y_true, Y_pred)
        return {"accuracy": acc, "f1_micro": f1, "hamming_loss": hamming}
