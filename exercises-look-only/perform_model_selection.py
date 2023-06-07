from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_ridge = np.linspace(.005, 1, n_evaluations)
    lam_lasso = np.linspace(.005, 2, n_evaluations)
    n_folds = 5
    ridge_validate = np.zeros((n_evaluations, 2))
    lasso_validate = np.zeros((n_evaluations, 2))
    for i, ridge in enumerate(lam_ridge):
        ridge_validate[i] = cross_validate(RidgeRegression(ridge), train_X, train_y, mean_square_error)
    for i, lasso in enumerate(lam_lasso):
        lasso_validate[i] = cross_validate(Lasso(lasso, max_iter=500), train_X, train_y, mean_square_error)
    fig = make_subplots(1, 2, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"], shared_xaxes=True)\
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$", width=750, height=300)\
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$")\
        .add_traces([go.Scatter(x=lam_ridge, y=ridge_validate[:, 0], name="Ridge Train Error"),
                    go.Scatter(x=lam_ridge, y=ridge_validate[:, 1], name="Ridge Validation Error"),
                    go.Scatter(x=lam_lasso, y=lasso_validate[:, 0], name="Lasso Train Error"),
                    go.Scatter(x=lam_lasso, y=lasso_validate[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    fig.show()
    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
    raise NotImplementedError()
