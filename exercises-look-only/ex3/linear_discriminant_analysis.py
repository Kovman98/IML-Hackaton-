from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        # """

        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.mu_ = np.zeros((n_classes, n_features))
        self.cov_ = np.identity(n_features)
        self.pi_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mu_[i] = np.mean(X_c, axis=0)
            self.cov_ += np.cov(X_c.T)
            self.pi_[i] = len(X_c)/len(X)

        self.cov_ /= n_classes
        self.cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return self.classes_[np.argmax(self.likelihood(X), axis = 1)]


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        features = X.shape[1]
        n_samples = X.shape[0]
        n_classes = self.mu_.shape[0]
        likelihood = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            diff = X - self.mu_[i]
            exponent = -1 / 2 * np.sum(diff @ self.cov_inv * diff, axis=1)
            denominator = np.sqrt((2 * np.pi) ** features * np.linalg.det(self.cov_))
            likelihood[:, i] = self.pi_[i] * np.exp(exponent) / denominator
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
