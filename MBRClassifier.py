import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.utils import check_X_y, check_array


class MBRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: object = None, meta_estimator: object = None):
        """
        Parameters
        ----------
        base_estimator : scikit-learn binary classifier, default=LogisticRegression()
            The classifier used in the first (BR) stage.
        meta_estimator : scikit-learn binary classifier, default=LogisticRegression()
            The classifier used in the the second (meta) stage.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression()
        if meta_estimator is None:
            meta_estimator = LogisticRegression()
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator

    def fit(self, X: numpy.ndarray, Y: numpy.ndarray):
        """
        Fit the Meta Binary Relevance classifier.

        This method trains |L| binary classifiers in the first stage (BR) using the entire training set.
        Then, it obtains first-stage predictions to augment the original feature space, and trains |L|
        meta classifiers on the extended features.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The training input samples.
        Y : numpy.ndarray of shape (n_samples, n_labels)
            The binary indicator matrix for labels.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate inputs and convert Y to numpy array
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = numpy.array(Y)
        n_samples, n_labels = Y.shape
        self.n_labels_ = n_labels

        # ------------------------
        # First Stage: Binary Relevance
        # ------------------------
        self.first_stage_classifiers_ = []
        for i in range(n_labels):
            clf = clone(self.base_estimator)
            clf.fit(X, Y[:, i])
            self.first_stage_classifiers_.append(clf)

        # Generate first-stage predictions on training data.
        # We assume that the estimator has a predict_proba method.
        first_stage_preds = numpy.zeros((n_samples, n_labels))
        for i in range(n_labels):
            # Use probability for the positive class
            first_stage_preds[:, i] = self.first_stage_classifiers_[i].predict_proba(X)[:, 1]

        # Augment the original features with the first-stage predictions.
        X_meta = numpy.hstack((X, first_stage_preds))

        # ------------------------
        # Second Stage: Meta-Level Binary Relevance
        # ------------------------
        self.meta_classifiers_ = []
        for i in range(n_labels):
            clf = clone(self.meta_estimator)
            clf.fit(X_meta, Y[:, i])
            self.meta_classifiers_.append(clf)

        return self

    def predict_proba(self, X: numpy.ndarray):
        """
        Predict probability estimates for each label.

        First, the first-stage classifiers are applied to X to generate predictions.
        These predictions augment the original feature space to form X_meta.
        Then, the meta classifiers provide the final probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_labels)
            Probability estimates for each label.
        """
        X = check_array(X)
        n_samples = X.shape[0]

        # First-stage predictions on new data.
        first_stage_preds = numpy.zeros((n_samples, self.n_labels_))
        for i in range(self.n_labels_):
            first_stage_preds[:, i] = self.first_stage_classifiers_[i].predict_proba(X)[:, 1]

        # Augment original features with first-stage predictions.
        X_meta = numpy.hstack((X, first_stage_preds))

        # Second-stage (meta) predictions.
        proba = numpy.zeros((n_samples, self.n_labels_))
        for i in range(self.n_labels_):
            proba[:, i] = self.meta_classifiers_[i].predict_proba(X_meta)[:, 1]
        return proba

    def predict(self, X: numpy.ndarray):
        """
        Predict the multi-label outputs for each instance.

        Labels are predicted as relevant (1) if the estimated probability exceeds 0.5.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples, n_labels)
            Binary predictions for each label.
        """
        proba = self.predict_proba(X)
        predictions = (proba > 0.5).astype(int)
        return predictions

    def evaluate(self, X: numpy.ndarray, Y_true: numpy.ndarray):
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
