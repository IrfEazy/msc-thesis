import numpy
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.utils.validation import check_array, check_X_y


class PStClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, pruning_value=2, max_reintroduced=1):
        """
        Initialize the Pruned Sets classifier.

        Parameters
        ----------
        base_estimator : classifier object, default=LogisticRegression(max_iter=1000)
            The multi-class classifier used on the transformed problem.
        pruning_value : int, default=2
            The minimum frequency required for a label-set to be considered frequent.
        max_reintroduced : int, default=1
            The maximal number of frequent label-sets to reintroduce per infrequent example.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression(max_iter=1000)
        self.base_estimator = base_estimator
        self.pruning_value = pruning_value
        self.max_reintroduced = max_reintroduced

    def fit(self, X, Y):
        """
        Fit the Pruned Sets classifier.

        The training process includes:
          1. Counting label-set frequencies and determining which are frequent.
          2. Pruning infrequent label-sets.
          3. For each infrequent example, generating candidate frequent subsets (up to max_reintroduced per example)
             and adding them to the training set.
          4. Transforming the multi-label problem into a multi-class problem and training the classifier.

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

        # Convert each label vector to a tuple to serve as a dictionary key.
        label_tuples = [tuple(row) for row in Y]

        # Count frequency of each label-set.
        freq_dict = {}
        for lt in label_tuples:
            freq_dict[lt] = freq_dict.get(lt, 0) + 1

        # Identify frequent label-sets.
        frequent_label_sets = {lt for lt, count in freq_dict.items() if count >= self.pruning_value}

        # Build new training data: (X_new, Y_new)
        X_new = []
        Y_new = []

        for i, lt in enumerate(label_tuples):
            if lt in frequent_label_sets:
                # Keep examples with frequent label-sets.
                X_new.append(X[i])
                Y_new.append(lt)
            else:
                # For infrequent examples, attempt to reintroduce by generating frequent subsets.
                # Identify indices with a positive label.
                positive_indices = [j for j, val in enumerate(lt) if val == 1]
                candidate_subsets = set()
                # Generate all proper non-empty subsets.
                for r in range(1, len(positive_indices)):
                    for subset in combinations(positive_indices, r):
                        candidate = [0] * self.n_labels_
                        for j in subset:
                            candidate[j] = 1
                        candidate_subsets.add(tuple(candidate))
                # Retain only those candidate subsets that are frequent.
                frequent_candidates = [cand for cand in candidate_subsets if cand in frequent_label_sets]
                # Reintroduce up to max_reintroduced examples from these candidates.
                count = 0
                for cand in frequent_candidates:
                    if count >= self.max_reintroduced:
                        break
                    X_new.append(X[i])
                    Y_new.append(cand)
                    count += 1

        X_new = numpy.array(X_new)
        # Y_new is a list of tuples.

        # Map each unique label-set in Y_new to a unique class index.
        unique_label_sets = list(set(Y_new))
        self.label_to_class_ = {lt: idx for idx, lt in enumerate(unique_label_sets)}
        self.class_to_label_ = {idx: numpy.array(lt) for lt, idx in self.label_to_class_.items()}

        # Transform multi-label targets into multi-class targets.
        Y_transformed = numpy.array([self.label_to_class_[lt] for lt in Y_new])

        # Train the multi-class classifier.
        self.classifier_ = clone(self.base_estimator)
        self.classifier_.fit(X_new, Y_transformed)

        # Precompute the label matrix for each class (used in predict_proba).
        n_classes = len(self.class_to_label_)
        self.class_label_matrix_ = numpy.zeros((n_classes, self.n_labels_), dtype=int)
        for class_idx, label_arr in self.class_to_label_.items():
            self.class_label_matrix_[class_idx, :] = label_arr

        return self

    def predict(self, X):
        """
        Predict multi-label outputs for the given data.

        Each instance is assigned a class by the multi-class classifier, and this class is mapped
        back to its corresponding label-set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.

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

        The multi-class classifier outputs probabilities for each class.
        Marginal probability for a label is computed by summing the probabilities of all classes
        whose associated label-set has that label present.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.

        Returns
        -------
        Y_proba : ndarray of shape (n_samples, n_labels)
            The estimated probabilities for each label.
        """
        X = check_array(X)
        proba = self.classifier_.predict_proba(X)  # shape: (n_samples, n_classes)
        Y_proba = numpy.dot(proba, self.class_label_matrix_)
        return Y_proba

    def evaluate(self, X, Y_true):
        """
        Evaluate the Pruned Sets classifier using standard multi-label metrics.

        Metrics computed include:
          - Subset Accuracy
          - Hamming Loss
          - Micro-averaged F1 Score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.
        Y_true : array-like of shape (n_samples, n_labels)
            The true binary label matrix.

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
