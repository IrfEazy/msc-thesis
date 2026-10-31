import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import fit_binary, positive_score, resolve_estimator
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class CDNClassifier(BaseEstimator, ClassifierMixin):
    """Conditional Dependency Network (Guo & Gu 2011), inference by Gibbs sampling.

    One model per label is trained on ``[X, other true labels]``. Prediction
    runs a Gibbs sampler over the label vector of every instance and estimates
    the marginals as the average of the samples kept after burn-in.

    Parameters
    ----------
    base_estimator : estimator, default None
        Binary classifier cloned for each label. None means a fresh
        ``LogisticRegression(max_iter=1000)``.
    n_iterations : int, default 100
        Total number of Gibbs sweeps (each sweep updates every label once).
    burn_in : int, default 50
        Initial sweeps discarded; must satisfy ``0 <= burn_in < n_iterations``.
    random_state : int or None, default None
        Seed of the sampler. A single generator is created for each
        ``predict_proba`` call, so a fixed seed gives reproducible output.

    Attributes
    ----------
    estimators_ : list of fitted binary classifiers, one per label
        A label with a constant training column gets a constant predictor.
    n_labels_ : int
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        n_iterations: int = 100,
        burn_in: int = 50,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.random_state = random_state

    def _check_sampling(self):
        if self.burn_in < 0 or self.burn_in >= self.n_iterations:
            raise ValueError(
                "burn_in must satisfy 0 <= burn_in < n_iterations, got "
                f"burn_in={self.burn_in}, n_iterations={self.n_iterations}."
            )

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "CDNClassifier":
        """
        Fit one binary classifier per label on the features augmented with
        all the other true labels.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training input samples.
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary indicator matrix of labels.

        Returns
        -------
        self : "CDNClassifier"
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``burn_in`` is negative or not smaller than ``n_iterations``.
        """
        self._check_sampling()
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        template = resolve_estimator(self.base_estimator)
        self.estimators_ = [
            fit_binary(template, numpy.hstack((X, numpy.delete(Y, j, axis=1))), Y[:, j])
            for j in range(Y.shape[1])
        ]
        self.n_labels_ = Y.shape[1]
        return self

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Estimate the label marginals by Gibbs sampling.

        The sweeps are vectorised across instances; each sweep visits the
        labels in a fresh random order. The values are sample frequencies
        (averages of 0/1 draws after burn-in), not calibrated probabilities.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_labels)
        """
        self._check_sampling()
        X = numpy.asarray(X)
        n, q = X.shape[0], self.n_labels_
        rng = numpy.random.RandomState(self.random_state)
        state = rng.randint(0, 2, size=(n, q))
        total = numpy.zeros((n, q))
        p_feat = X.shape[1]
        # Estimator j sees [X, all labels but j]; the buffer is allocated once
        # and its label columns are overwritten in place at every step.
        X_aug = numpy.empty((n, p_feat + q - 1))
        X_aug[:, :p_feat] = X
        for iteration in range(self.n_iterations):
            for j in rng.permutation(q):
                X_aug[:, p_feat : p_feat + j] = state[:, :j]
                X_aug[:, p_feat + j :] = state[:, j + 1 :]
                p = positive_score(self.estimators_[j], X_aug)
                state[:, j] = rng.random_sample(n) < p
            if iteration >= self.burn_in:
                total += state
        return total / (self.n_iterations - self.burn_in)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> numpy.ndarray:
        """
        Predict labels: 1 where the estimated marginal exceeds ``threshold``.

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
        """
        return (self.predict_proba(X) > threshold).astype(int)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(
        self, X: ArrayLike, Y: ArrayLike, threshold: float = 0.5
    ) -> dict[str, float]:
        """
        Evaluate the classifier.

        Returns
        -------
        metrics : dict
            See ``functions.assess``; AUC uses ``predict_proba`` scores.
        """
        proba = self.predict_proba(X)
        return assess(Y, (proba > threshold).astype(int), proba)
