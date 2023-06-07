from typing import Tuple
import numpy as np
import pandas as pd
import agoda_preprocess
import sklearn
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """

    train_x: pd.DataFrame = X.sample(frac=train_proportion)
    train_y: pd.Series = y.loc[train_x.index]
    test_x: pd.DataFrame = X.loc[X.index.difference(train_x.index)]
    test_y: pd.Series = y.loc[test_x.index]

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # loading the data and shuffle it
    dataFrame: pd.DataFrame = pd.read_csv("dataset/agoda_cancellation_train.csv")
    shuffled_dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

    # splitting into X and y
    y: pd.Series = dataFrame["cancellation_datetime"]
    y = pd.Series(np.where(y == 0, 0, 1))
    X: pd.DataFrame = dataFrame.drop('cancellation_datetime', axis=1)

    train_x, train_y, test_x, test_y = split_train_test(dataFrame, y)


    # corolated = X["customer_nationality"]
    # label = X[""]
    # for label in X:
    #     rho = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

    fitted: LinearRegression = LinearRegression().fit(train_x, train_y)

    # mse = metrics.mean_squared_error(y, fitted.predict(test_x))
    # print("Mean Squared Error:", mse)




