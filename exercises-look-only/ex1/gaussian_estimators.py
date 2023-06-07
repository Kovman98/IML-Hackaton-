from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        n_samples = len(X)-1
        if self.biased_:
            n_samples += 1
        sum_samples = 0
        sum_samples_squared = 0
        for sample in X:
            sum_samples += sample
            sum_samples_squared += (sample**2)
        expectation_squared = sum_samples_squared/n_samples
        self.mu_ = sum_samples/n_samples
        self.var_ = expectation_squared-(self.mu_**2)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf = np.exp(-(X-self.mu_)**2/(2*self.var_))/(np.sqrt(self.var_*2*np.pi))
        return pdf

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        return -np.sum((X - mu) ** 2) / (2 * sigma) - np.log(2 * np.pi * sigma) * (len(X) / 2)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        n_features = len(X[0])
        n_samples = len(X)
        sum_samples = []
        for feature in range(n_features):
            sum_samples.append(0)
            for sample in range(n_samples):
                sum_samples[feature] += X[sample][feature]
            sum_samples[feature] = sum_samples[feature]/n_samples
        self.mu_ = sum_samples
        X_centered = np.array((self.mu_)*n_samples).reshape(n_samples,n_features)
        X_centered = X-X_centered
        self.cov_ = 1/(n_samples-1)*X_centered.transpose().dot(X_centered)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        d = len(X[0])
        exp_term = -0.5*(X-self.mu_).transpose().dot(np.invert(self.cov_)).dot(X-self.mu_)
        pdf = 1/np.sqrt(((2*np.pi)**d)*det(self.cov_))*np.exp(exp_term)
        return pdf

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        # """
        n_samples=len(X)
        n_features = len(mu)
        log_likelihood = -(n_samples/2)*np.log((2*np.pi)**n_features)-n_samples/2*np.log(det(cov))
        X_centered = np.array([mu,]*n_samples)
        X_centered = X-X_centered
        sum = 0
        for i in range(n_samples):
            sum+= (X_centered[i,:].transpose().dot(inv(cov)).dot(X_centered[i,:]))
        log_likelihood += -0.5*sum
        return log_likelihood