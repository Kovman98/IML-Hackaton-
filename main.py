import sys
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
import task_1, task_2, agoda_preprocess
import plotly.graph_objects as go
from scipy.stats import pearsonr
from plotly.subplots import make_subplots




def find_best_threshold_linear(X, y, model):
    # Initialize LinearRegression model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform cross-validation to find the best threshold
    best_threshold = None
    best_accuracy = 0.0

    thershold_values = np.linspace(0, 1, 10)

    # Perform cross-validation and iterate over train/validation indices
    for value in thershold_values:
        model.fit(X_train, y_train)
        predicted_probs = expit(model.predict(X_test))
        predictions = np.where(predicted_probs >= value, 1, 0)
        accuracy = accuracy_score(y_test, predictions)
        if accuracy > best_accuracy:
            best_threshold = value
            best_accuracy = accuracy

    return best_threshold


def find_best_alpha(model, X, y, threshold = 0.66):
    # Define a range of alpha values to test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform cross-validation to find the best threshold
    best_alpha = None
    best_accuracy = 0.0

    alpha_values = np.linspace(1e-3, 0.5, 10)

    # Perform cross-validation and iterate over train/validation indices
    for alpha in alpha_values:
        if (model == "ridge"):
            model: Ridge = Ridge(alpha=alpha)
        else:
            model: Lasso = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        predicted_probs = expit(model.predict(X_test))
        predictions = np.where(predicted_probs >= threshold, 1, 0)
        accuracy = accuracy_score(y_test, predictions)
        if accuracy > best_accuracy:
            best_alpha = alpha
            best_accuracy = accuracy

    return best_alpha


def load_data(path: str) -> pd.DataFrame:
    dataFrame: pd.DataFrame = pd.read_csv(path, parse_dates=['booking_datetime', 'checkout_date', 'cancellation_datetime'],
                                                dayfirst=True)
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    dataFrame['checkin_date'] = pd.to_datetime(dataFrame['checkin_date'], format='%d/%m/%Y %H:%M')

    dataFrame['is_cancelled'] = dataFrame['cancellation_datetime'].fillna(0)
    dataFrame['is_cancelled'].loc[dataFrame['is_cancelled'] != 0] = 1
    return dataFrame


def split_train_test(df: pd.DataFrame, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:

    train: pd.DataFrame = df.sample(frac=train_proportion)
    test: pd.DataFrame = df.loc[df.index.difference(train.index)]

    return train, test


def model_selection(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    # tree model
    best_depth = 99
    best_accuracy = 1
    for i in range(2, 5):
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
    # best_threshold_linear = find_best_threshold_linear(train_x, train_y, LinearRegression())
    best_threshold_linear = 0.61
    ridge_model: Ridge = Ridge()
    lasso_model: Lasso = Lasso()
    best_alpha_ridge = find_best_alpha("ridge", train_x, train_y)
    best_alpha_lasso = find_best_alpha("lasso", train_x, train_y)

    # fitting the models with the best values
    # best_threshold_rf = find_best_threshold_linear(train_x, train_y, RandomForestRegressor())
    best_threshold_rf = 0.66
    rf_model = RandomForestRegressor().fit(train_x, train_y)
    linear_pred: LinearRegression = LinearRegression().fit(train_x, train_y)
    best_ridge: Ridge = Ridge(best_alpha_ridge).fit(train_x, train_y)
    best_lasso: Lasso = Lasso(best_alpha_lasso).fit(train_x, train_y)
    logistic_model = LogisticRegression().fit(train_x, train_y)

    # getting the best prediction of all models
    predicted_probs = expit(linear_pred.predict(test_x))
    predictions = np.where(predicted_probs >= best_threshold_linear, 1, 0)
    predicted_probs_lasso = expit(best_lasso.predict(test_x))
    predictions_lasso = np.where(predicted_probs_lasso >= best_threshold_linear, 1, 0)
    predicted_probs_ridge = expit(best_ridge.predict(test_x))
    predictions_ridge = np.where(predicted_probs_ridge >= best_threshold_linear, 1, 0)

    logistic_pred = logistic_model.predict(test_x)
    rf_predict = rf_model.predict(test_x)
    rf_predicted_probs = expit(rf_predict)
    rf_predictions = np.where(rf_predicted_probs >= best_threshold_rf, 1, 0)

    # calculating accuracy scores for all models
    accuracy_linear: float = accuracy_score(test_y, predictions)
    accuracy_ridge: float = accuracy_score(test_y, predictions_ridge)
    accuracy_lasso: float = accuracy_score(test_y, predictions_lasso)
    accuracy_logistic: float = accuracy_score(test_y, logistic_pred)
    accuracy_rf: float = accuracy_score(test_y, rf_predictions)

    print("Accuracy of Linear Regression:", accuracy_linear)
    print("Accuracy of Ridge Regression:", accuracy_ridge)
    print("Accuracy of Lasso Regression:", accuracy_lasso)
    print("Accuracy of random forest:", accuracy_rf)
    print("Accuracy of Logistic Regression:", accuracy_logistic)
    # Calculate the F1 macro score for each model
    f1_macro_linear = f1_score(test_y, predictions, average='macro')
    f1_macro_ridge = f1_score(test_y, predictions_ridge, average='macro')
    f1_macro_lasso = f1_score(test_y, predictions_lasso, average='macro')
    f1_macro_logistic = f1_score(test_y, logistic_pred, average='macro')
    f1_macro_rf = f1_score(test_y, rf_predictions, average='macro')

    # Print the F1 macro scores
    print("F1 Macro Score - Linear:", f1_macro_linear)
    print("F1 Macro Score - Ridge:", f1_macro_ridge)
    print("F1 Macro Score - Lasso:", f1_macro_lasso)
    print("F1 Macro Score - Logistic:", f1_macro_logistic)
    print("F1 Macro Score - Random Forest:", f1_macro_rf)


def clusterin(X):
    train, test = split_train_test(X)
    y_corr = pd.DataFrame()
    y_corr['is_cancelled'] = test['is_cancelled'].astype(int)
    train_x = train.drop('is_cancelled', axis = 1)
    test_x = test.drop('is_cancelled', axis = 1)
    num_clusters = []
    cancel_prob = []

    best_num_cluster = None
    best_correlation = -1

    fig = make_subplots(rows=3, cols=3,
                        subplot_titles=[f"Number of Clusters: {num_cluster}" for num_cluster in range(3, 10)])

    for i, num_cluster in enumerate(range(3, 10)):
        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(train_x)
        prediction = kmeans.predict(test_x)
        y_corr['cluster_label'] = prediction
        cluster_cancel_prob = y_corr.groupby('cluster_label')['is_cancelled'].mean()

        fig.add_trace(go.Scatter(x=cluster_cancel_prob.index, y=cluster_cancel_prob.values, mode='lines+markers'),
                      row=(i // 3) + 1, col=(i % 3) + 1)

    fig.update_layout(title='Cluster Cancellation Probability for Different Numbers of Clusters')
    fig.update_xaxes(title_text='Cluster Num')
    fig.update_yaxes(title_text='Cancellation Probability')
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)

    if len(sys.argv) != 3:
        print("Unexpected number of arguments")
    else:
        path1: str = sys.argv[1]
        path2: str = sys.argv[2]

        # loading the data
        dataFrame: pd.DataFrame = load_data("dataset/agoda_cancellation_train.csv")

        # preprocess all data and splitting into test and train sets:
        dataFrame = agoda_preprocess.preprocess_data(dataFrame)
        dataFrame['is_cancelled'] = dataFrame['cancellation_datetime'].fillna(0)
        dataFrame['is_cancelled'].loc[dataFrame['is_cancelled'] != 0] = 1
        train, test = split_train_test(dataFrame)

        clean_train = agoda_preprocess.preprocess_train(train)
        mean_features_values: agoda_preprocess.MeanValuesCalculator = agoda_preprocess.MeanValuesCalculator(train)
        clean_test = agoda_preprocess.preprocess_test(test, mean_features_values.mean_values)
        clean_test = clean_test.drop(['cancellation_datetime'], axis=1)

        # splitting train and test sets into x and y
        train_y = clean_train["is_cancelled"].astype(int)
        train_x = clean_train.drop('is_cancelled', axis=1)
        train_x = train_x.drop('h_booking_id', axis=1)
        test_y = clean_test["is_cancelled"].astype(int)
        test_x = clean_test.drop('is_cancelled', axis=1)
        test_x = test_x.drop('h_booking_id', axis=1)

        # model selecting part:
        # model_selection(train_x, train_y, test_x, test_y)

        # task 1:
        best_model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1).fit(train_x, train_y)
        # task_1.task_1(best_model, path1, mean_features_values.mean_values)

        # question 2:
        train_x = train_x.drop('original_selling_amount', axis=1)
        best_model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1).fit(train_x, train_y)

        task_2.task_2(best_model, path2, mean_features_values, clean_train, clean_test)

        #question 3:

        features = ["number_of_nights", "days_before_checkin", "hotel_star_rating", "no_of_adults",
                    "no_of_children", "no_of_extra_bed", "no_of_room", "original_selling_amount",
                    "cancel_for_free", "did_request", "is_first_booking", "is_user_logged_in"]

        correlations = []

        for feature in features:
            correlation = np.cov(train_x[feature], train_y)[0, 1] / (np.std(train_x[feature]) * np.std(train_y))
            correlations.append(correlation)

        # Find the index of the feature with the highest correlation
        max_corr_index = np.argmax(np.abs(correlations))

        # # Create a color list for the bar plot
        colors = ['#ffcccc'] * len(features)  # Default color for all features
        colors[max_corr_index] = '#ff0000'  # Color for the feature with highest correlation

        # # Create the bar plot using Plotly
        fig = go.Figure(data=go.Bar(x=features, y=correlations, marker=dict(color=colors)))

        # # Customize the layout of the plot
        fig.update_layout(title='Correlations between Features and Target Variable',
                          xaxis_title='Features',
                          yaxis_title='Correlation',
                          plot_bgcolor='rgba(0, 0, 0, 0)')  # Set plot background color to transparent

        # Show the plot
        fig.show()

        #question 4:
        clusterin(clean_train)
