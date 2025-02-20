import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.utils.validation import check_array, check_X_y


class LPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        """
        Initialize the Label Powerset classifier.

        Parameters
        ----------
        base_estimator : classifier object, default=LogisticRegression(max_iter=1000)
            The multi-class classifier to be used on the transformed problem.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression(max_iter=1000)
        self.base_estimator = base_estimator

    def fit(self, X, Y):
        """
        Fit the Label Powerset classifier.

        The multi-label training set is transformed into a multi-class training set by mapping
        each unique label set to a unique class. A multi-class classifier is then trained on this data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.
        Y : array-like of shape (n_samples, n_labels)
            The binary indicator matrix for labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = numpy.array(Y)
        self.n_labels_ = Y.shape[1]

        # Map each label vector to a tuple so it can be used as a dict key.
        label_tuples = [tuple(row) for row in Y]
        # Create a mapping: unique label-set tuple -> unique class index.
        unique_label_tuples = list(set(label_tuples))
        self.label_to_class_ = {label_tuple: idx for idx, label_tuple in enumerate(unique_label_tuples)}
        self.class_to_label_ = {idx: numpy.array(label_tuple) for label_tuple, idx in self.label_to_class_.items()}

        # Transform multi-label targets into a single multi-class target.
        Y_transformed = numpy.array([self.label_to_class_[tuple(row)] for row in Y])

        # Train the multi-class classifier.
        self.classifier_ = clone(self.base_estimator)
        self.classifier_.fit(X, Y_transformed)

        # Precompute a matrix of label sets for each class.
        n_classes = len(self.class_to_label_)
        self.class_label_matrix_ = numpy.zeros((n_classes, self.n_labels_), dtype=int)
        for class_idx, label_arr in self.class_to_label_.items():
            self.class_label_matrix_[class_idx, :] = label_arr

        return self

    def predict(self, X):
        """
        Predict multi-label outputs for the given data.

        For each instance, the multi-class classifier is used to predict a class,
        which is then mapped back to the corresponding multi-label set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_labels)
            The predicted binary label matrix.
        """
        X = check_array(X)
        Y_class_pred = self.classifier_.predict(X)
        Y_pred = numpy.array([self.class_to_label_[cls] for cls in Y_class_pred])
        return Y_pred

    def predict_proba(self, X):
        """
        Predict marginal probabilities for each label.

        For each instance, the multi-class classifier outputs probabilities for each class.
        The marginal probability for a given label is computed by summing the probabilities of
        all classes whose corresponding label set has that label present.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.


        Returns
        -------
        Y_proba : ndarray of shape (n_samples, n_labels)
            The estimated probability for each label.
        """

        X = check_array(X)
        # Get multi-class probability estimates.
        proba = self.classifier_.predict_proba(X)  # shape: (n_samples, n_classes)
        # Compute marginal probabilities: dot the probability matrix with the class label matrix.
        Y_proba = numpy.dot(proba, self.class_label_matrix_)
        return Y_proba

    def evaluate(self, X, Y_true):
        """
        Evaluate the Label Powerset classifier using standard multi-label metrics.

        The following metrics are computed:
          - Subset Accuracy
          - Hamming Loss
          - Micro-averaged F1 Score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Y_true : array-like of shape (n_samples, n_labels)
            The true binary indicator matrix for labels.

        Returns
        -------
        metrics : dict
            A dictionary containing accuracy, hamming loss, and micro F1 score.
        """
        Y_true = numpy.array(Y_true)
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y_true, Y_pred)
        hamming = hamming_loss(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average='micro')
        return {"accuracy": accuracy, "hamming_loss": hamming, "f1_micro": f1}
