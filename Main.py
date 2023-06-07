from typing import Tuple
import numpy as np
import pandas as pd
import agoda_preprocess
import sklearn
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def find_best_alpha_threshold(model, X_train, y_train, X_val, y_val, alpha_values, threshold_values):
    best_accuracy = 0.0
    best_alpha = None
    best_threshold = None

    for alpha in alpha_values:
        for threshold in threshold_values:
            # Train the regression model with the current alpha value
            model.alpha = alpha
            model.fit(X_train, y_train)

            # Obtain predictions on the validation data and apply sigmoid function
            X_val_pred = model.predict(X_val)
            y_val_pred = sigmoid(X_val_pred) >= threshold

            # Evaluate the performance using accuracy score
            accuracy = accuracy_score(y_val, y_val_pred)

            # Check if current combination is the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha
                best_threshold = threshold

    return best_alpha, best_threshold, best_accuracy


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    train_x['cancellation_datetime'] = train_y
    test_x: pd.DataFrame = X.loc[X.index.difference(train_x.index)]
    test_y: pd.Series = y.loc[test_x.index]
    test_x['cancellation_datetime'] = test_y

    return train_x, test_x


if __name__ == '__main__':
    # loading the data and shuffling it
    dataFrame: pd.DataFrame = pd.read_csv("dataset/agoda_cancellation_train.csv")
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

    # splitting into X and y and into train and test
    y: pd.Series = dataFrame["cancellation_datetime"]
    y = pd.Series(np.where(y == 0, 0, 1))
    X: pd.DataFrame = dataFrame.drop('cancellation_datetime', axis=1)

    train, test = split_train_test(dataFrame, y)
    # preprocess on train:

    # preprocess on test:


    # deviation into mini train and validation sets
    train_y = train["cancellation_datetime"]
    train_x = train.drop('cancellation_datetime', axis=1)
    train_smaller, validation = split_train_test(train_x, train_y)

    train_smaller_x = train_validation["cancellation_datetime"]
    train_smaller_y = train_validation.drop('cancellation_datetime', axis=1)
    validation_x = test_validation["cancellation_datetime"]
    validation_y = test_validation.drop('cancellation_datetime', axis=1)

    # tree model
    tree_model = XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1)  # todo max_depth should be chosen in for loop
    scores = cross_val_score(tree_model, X_train, y_train, cv=5, scoring='accuracy')
    average_accuracy = np.mean(scores)


    # linear models
    fitted_linear: LinearRegression = LinearRegression().fit(train_x, train_y)

    # todo should run in for loop for choosing the regularization param and the threshold param
    fitted_ridge: Ridge = Ridge().fit(train_x, train_y)
    fitted_lasso: Lasso = Lasso().fit(train_x, train_y)



    # ridge_model = Ridge(alpha=0.1)
    # ridge_model.fit(X_train, y_train)
    # lasso_model = Lasso(alpha=0.1)
    # lasso_model.fit(X_train,y_train)
    # # mse = metrics.mean_squared_error(y, fitted.predict(test_x))
    # # print("Mean Squared Error:", mse)




