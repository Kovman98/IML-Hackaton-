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
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import matplotlib.pyplot as plt


def find_best_threshold_linear(X, y):
    # Initialize LinearRegression model
    model = LinearRegression()

    # Perform cross-validation to find the best threshold
    best_threshold = None
    best_accuracy = 0.0

    for train_index, val_index in cross_val_score(model, X, y, cv=5):
        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred_val = model.predict(X_val)

        # Apply sigmoid function to convert predictions into probabilities
        y_pred_prob_val = expit(y_pred_val)

        # Define a range of thresholds to test
        thresholds = np.arange(0.1, 1.0, 0.1)

        for threshold in thresholds:
            # Convert probabilities to binary labels based on the threshold
            y_pred_labels_val = np.where(y_pred_prob_val >= threshold, 1, 0)

            # Calculate accuracy using the predicted labels and the true labels
            accuracy = accuracy_score(y_val, y_pred_labels_val)

            # Check if the current threshold gives higher accuracy
            if accuracy > best_accuracy:
                best_threshold = threshold
                best_accuracy = accuracy

    return best_threshold


def find_best_alpha_threshold(model, X, y):
    # Define a range of alpha values to test
    alphas = np.arange(0.1, 1.0, 0.1)

    # Perform cross-validation to find the best alpha
    best_alpha = None
    best_alpha_accuracy = 0.0
    best_threshold = None

    for alpha in alphas:
        # Set the current alpha value
        model.alpha = alpha

        accuracy_scores = []
        thresholds = np.arange(0.1, 1.0, 0.1)

        # Iterate over each fold of cross-validation
        for train_index, val_index in cross_val_score(model, X, y, cv=5):
            # Split data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Fit the model on the training set
            model.fit(X_train, y_train)

            # Make predictions on the validation set
            y_pred_val = model.predict(X_val)

            # Apply sigmoid function to convert predictions into probabilities
            y_pred_prob_val = expit(y_pred_val)

            for threshold in thresholds:
                # Convert probabilities to binary labels based on the threshold
                y_pred_labels_val = np.where(y_pred_prob_val >= threshold, 1, 0)

                # Calculate accuracy using the predicted labels and the true labels
                accuracy = accuracy_score(y_val, y_pred_labels_val)

                accuracy_scores.append(accuracy)

        # Calculate the average accuracy across all folds
        average_accuracy = np.mean(accuracy_scores)

        # Check if the current alpha gives higher accuracy
        if average_accuracy > best_alpha_accuracy:
            best_alpha = alpha
            best_alpha_accuracy = average_accuracy
            best_threshold = np.argmax(accuracy_scores) // len(thresholds)

    return best_alpha,best_threshold


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


def model_selection(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    # tree model
    best_depth = 99
    best_accuracy = 1
    for i in range(2, 11):
        tree_model = XGBClassifier(n_estimators=100, max_depth=i, learning_rate=0.1)
        scores = cross_val_score(tree_model, train_x, train_y, cv=5, scoring='accuracy')
        average_accuracy = np.mean(scores)

        if average_accuracy < best_accuracy:
            best_accuracy = average_accuracy
            best_depth = i

    print("Best accuracy of tree is: ", best_accuracy)
    print("The best tree depth is: ", best_depth)

    # linear models
    # getting best threshold and best alpha
    best_threshold_linear = find_best_threshold_linear(train_x, train_y)
    ridge_model: Ridge = Ridge()
    lasso_model: Lasso = Lasso()
    best_alpha_ridge, best_threshold_ridge = find_best_alpha_threshold(ridge_model,train_x, train_y)
    best_alpha_lasso, best_threshold_lasso = find_best_alpha_threshold(lasso_model,train_x, train_y)

    # fitting the models with the best values
    linear_pred: LinearRegression = LinearRegression().fit(train_x, train_y)
    best_ridge: Ridge = Ridge(best_alpha_ridge).fit(train_x, train_y)
    best_lasso: Lasso = Lasso(best_alpha_lasso).fit(train_x, train_y)
    logistic_model = LogisticRegression().fit(train_x, train_y)

    # getting the best prediction of all models
    linear_pred = linear_pred.predict(test_x).apply(lambda x: 1 if x >= best_threshold_linear else 0)
    ridge_pred = best_ridge.predict(test_x).apply(lambda x: 1 if x >= best_threshold_ridge else 0)
    lasso_pred = best_lasso.predict(test_x).apply(lambda x: 1 if x >= best_threshold_lasso else 0)
    logistic_pred = best_lasso.predict(test_x)

    # calculating accuracy scores for all models
    accuracy_linear: float = accuracy_score(test_y, linear_pred)
    accuracy_ridge: float = accuracy_score(test_y, ridge_pred)
    accuracy_lasso: float = accuracy_score(test_y, lasso_pred)
    accuracy_logistic: float = accuracy_score(test_y, logistic_pred)

    print("Accuracy of Linear Regression:", accuracy_linear)
    print("Accuracy of Ridge Regression:", accuracy_ridge)
    print("Accuracy of Lasso Regression:", accuracy_lasso)
    print("Accuracy of Logistic Regression:", accuracy_logistic)


if __name__ == '__main__':
    # loading the data and shuffling it
    dataFrame: pd.DataFrame = pd.read_csv("dataset/agoda_cancellation_train.csv")
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

    # splitting into X and y and into train and test
    y: pd.Series = dataFrame["cancellation_datetime"]
    y = pd.Series(np.where(y == 0, 0, 1))
    X: pd.DataFrame = dataFrame.drop('cancellation_datetime', axis=1)

    train, test = split_train_test(X, y)

    grouped_policies = train.groupby('cancellation_datetime')['cancellation_policy_code'].apply(list).to_dict()
    for count, policies in grouped_policies.items():
        # print(f'Group with {count} cancellation(s): {policies}')
        print(count)
    # policies_high_cancellation = cancellation_rates.sort_values(ascending=False).index
    # print(policies_high_cancellation)
    # threshold = 0.5  # Define your threshold here
    # train['Cluster'] = np.where(train['cancellation_policy_code'].isin(policies_high_cancellation), 'High Cancellation',
    #                          'Low Cancellation')
    # cluster_counts = train['Cluster'].value_counts()
    # plt.bar(cluster_counts.index, cluster_counts.values)
    # plt.xlabel('Cluster')
    # plt.ylabel('Count')
    # plt.title('Policy Clusters Based on Cancellation Likelihood')
    # plt.show()

    # preprocess on train:

    # preprocess on test:


    # splitting train and test sets into x and y
    train_y = train["cancellation_datetime"]
    train_x = train.drop('cancellation_datetime', axis=1)
    test_y = test["cancellation_datetime"]
    test_x = test.drop('cancellation_datetime', axis=1)





    # todo needed? maybe
    # deviation into mini train and validation sets
    # train_smaller, validation = split_train_test(train_x, train_y)
    #
    # train_smaller_x = train_smaller["cancellation_datetime"]
    # train_smaller_y = train_smaller.drop('cancellation_datetime', axis=1)
    # validation_x = validation["cancellation_datetime"]
    # validation_y = validation.drop('cancellation_datetime', axis=1)

    # model selecting:
    model_selection(train_x, train_y, test_x, test_y)



