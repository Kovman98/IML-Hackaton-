from typing import Tuple
import numpy as np
import pandas as pd
import agoda_preprocess
import sklearn
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold




def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    # dropping unneeded columns
    X = X.drop(['hotel_live_date', 'h_customer_id', 'customer_nationality', 'origin_country_code', 'language',
                'original_payment_currency', 'original_payment_method', 'hotel_brand_code'], axis=1)

    # creating new columns for needed info
    X['is_cancelled'] = X['cancellation_datetime'].fillna(0)
    X['is_cancelled'].loc[X['is_cancelled'] != 0] = 1

    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['number_of_nights'] = (X['checkout_date'] - X['checkin_date']).dt.days

    # preprocessing the cancellation policy
    X['cancellation_policy'] = X['cancellation_policy'].fillna('')
    X['cancel_for_free'] = 0  # default value

    for i, cancellation_policy in enumerate(X['cancellation_policy']):
        # need to check if inside policy
        if cancellation_policy != '':
            cancellation_parts = cancellation_policy.split('_')
            for part in cancellation_parts:
                if 'D' in part:
                    first, second = part.split('D')[0], part.split('D')[1]
                    if int(first) <= X['days_before_checkin'][i] and second[0] != '0':
                        X['cancel_for_free'][i] = 1

    # dropping columns that not needed (created alternative columns for these ones)
    X = X.drop(['checkin_date', 'checkout_date', 'cancellation_datetime', 'cancellation_policy'])

    X['is_user_logged_in'] = X['is_user_logged_in'].astype(bool).astype(int)
    X['is_first_booking'] = X['is_first_booking'].astype(bool).astype(int)

    # using get_dummies on values labels that their values has no numerical meaning
    X = pd.get_dummies(X, prefix='hotel_country_code_', columns=['hotel_country_code'])
    X = pd.get_dummies(X, prefix='accommadation_type_name_', columns=['accommadation_type_name'])
    X = pd.get_dummies(X, prefix='guest_nationality_country_name_', columns=['guest_nationality_country_name'])
    X = pd.get_dummies(X, prefix='original_payment_type_', columns=['original_payment_type'])
    X = pd.get_dummies(X, prefix='hotel_city_code_', columns=['hotel_city_code'])

    return X


def preprocess_train(X: pd.DataFrame) -> pd.DataFrame:
    # clearing duplicates
    X = X.dropna().drop_duplicates()

    # clearing noisy samples
    X = X[X["hotel_star_rating"].between(0, 5)]
    X = X[X["no_of_adults"].isin(range(12))]
    X = X[X["no_of_children"].isin(range(6))]
    X = X[X["no_of_extra_bed"].isin(range(3))]
    X = X[X["no_of_room"].isin(range(10))]
    X = X[X["original_selling_amount"].isin(range(6000))]
    X = X[X['number_of_nights'].isin(range(10))]
    X = X[X['days_before_checkin'].isin(range(200))]

    return X


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


def cancellation_cost(X,y,test_X, test_y):
    param_grid = {'alpha': np.logspace(-4, 4, 9)}  # Varying alpha values from 10^-4 to 10^4

    # Step 3: Perform cross-validated ridge regression
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)  # 5-fold cross-validation
    ridge_cv.fit(X, y)

    # Step 4: Get the best alpha and corresponding score for ridge regression
    best_alpha_ridge = ridge_cv.best_params_['alpha']

    # Step 5: Perform cross-validated lasso regression
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)  # 5-fold cross-validation
    lasso_cv.fit(X, y)

    # Step 6: Get the best alpha and corresponding score for lasso regression
    best_alpha_lasso = lasso_cv.best_params_['alpha']

    model_ridge = Ridge(best_alpha_ridge).fit(X, y)
    prediction_ridge = model_ridge.predict(test_X)
    model_lasso = Lasso(best_alpha_lasso).fit(X, y)
    prediction_lasso = model_lasso.predict(test_X)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, prediction_ridge))
    lasso_rmse = np.sqrt(mean_squared_error(y_test, prediction_lasso))

    # Print the RMSE scores
    print("Ridge RMSE:", ridge_rmse)
    print("Lasso RMSE:", lasso_rmse)

    if lasso_rmse < ridge_rmse:
        return model_ridge
    else:
        return model_lasso


if __name__ == '__main__':
    # question 1:

    # loading the data and shuffling it
    dataFrame: pd.DataFrame = pd.read_csv("dataset/agoda_cancellation_train.csv",
                                          parse_dates=['booking_datetime', 'checkout_date', 'cancellation_datetime'],
                                          dayfirst=True)
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    dataFrame['checkin_date'] = pd.to_datetime(dataFrame['checkin_date'], format='%d/%m/%Y %H:%M')

    # ids = dataFrame['h_booking_id']
    # dataFrame = dataFrame.drop('h_booking_id', axis=1)

    dataFrame['is_cancelled'] = dataFrame['cancellation_datetime'].fillna(0)
    dataFrame['is_cancelled'].loc[dataFrame['is_cancelled'] != 0] = 1
    # preprocess all data:
    dataFrame = agoda_preprocess.joint_preprocess(dataFrame)
    train, test = split_train_test(dataFrame)


    clean_train = agoda_preprocess.preprocess_train(train)
    clean_test = agoda_preprocess.preprocess_test(test)

    # todo preprocess on test: (in this part we need to fill the empty samples in the test)

    # splitting train and test sets into x and y
    train_y = clean_train["is_cancelled"].astype(int)
    train_x = clean_train.drop('is_cancelled', axis=1)
    train_x = train_x.drop('h_booking_id', axis=1)
    test_y = clean_test["is_cancelled"].astype(int)
    test_x = clean_test.drop('is_cancelled', axis=1)
    test_x = test_x.drop('h_booking_id', axis=1)


    # model selecting:
    # model_selection(train_x, train_y, test_x, test_y)


    # question 2:
    # Filter rows where 'cancellation_datetime' is not null
    filtered_train = clean_train[clean_train['is_cancelled'] == 1]
    filtered_test = clean_test[clean_test['is_cancelled'] == 1]

    # Create X (features) by excluding the specified columns
    X = filtered_train.drop(['is_cancelled', 'h_booking_id', 'original_selling_amount'], axis=1)
    X_test = filtered_test.drop(['is_cancelled', 'h_booking_id', 'original_selling_amount'], axis=1)

    # Create y (target variable) as 'original_selling_amount' column
    y = filtered_train['original_selling_amount']
    y_test = filtered_test['original_selling_amount']
    fitted_model = cancellation_cost(X, y, X_test, y_test)






    # todo needed? maybe
    # deviation into mini train and validation sets
    # train_smaller, validation = split_train_test(train_x, train_y)
    #
    # train_smaller_x = train_smaller["cancellation_datetime"]
    # train_smaller_y = train_smaller.drop('cancellation_datetime', axis=1)
    # validation_x = validation["cancellation_datetime"]
    # validation_y = validation.drop('cancellation_datetime', axis=1)


    # grouped_policies = train.groupby('cancellation_datetime')['cancellation_policy_code'].apply(list).to_dict()
    # for count, policies in grouped_policies.items():
    #     # print(f'Group with {count} cancellation(s): {policies}')
    #     print(count)
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