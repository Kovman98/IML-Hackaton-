import agoda_preprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def load_data(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path, parse_dates=['booking_datetime', 'checkout_date'], dayfirst=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df['checkin_date'] = pd.to_datetime(df['checkin_date'], format='%d/%m/%Y %H:%M')

    return df


def cancellation_cost_best_model(X, y, test_X, test_y):
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
    ridge_rmse = np.sqrt(mean_squared_error(test_y, prediction_ridge))
    lasso_rmse = np.sqrt(mean_squared_error(test_y, prediction_lasso))

    # Print the RMSE scores
    print("Ridge RMSE:", ridge_rmse)
    print("Lasso RMSE:", lasso_rmse)

    if lasso_rmse < ridge_rmse:
        return model_ridge
    else:
        return model_lasso


def task_2(best_model, path: str, mValues: pd.Series, clean_train: pd.DataFrame, clean_test: pd.DataFrame) -> None:
    # Filter rows where 'cancellation_datetime' is not null
    filtered_train = clean_train[clean_train['is_cancelled'] == 1]
    filtered_test = clean_test[clean_test['is_cancelled'] == 1]

    # Create X (features) by excluding the specified columns
    X = filtered_train.drop(['is_cancelled', 'h_booking_id', 'original_selling_amount'], axis=1)
    X_test = filtered_test.drop(['is_cancelled', 'h_booking_id', 'original_selling_amount'], axis=1)

    # Create y (target variable) as 'original_selling_amount' column
    y = filtered_train['original_selling_amount']
    y_test = filtered_test['original_selling_amount']

    # getting the best linear model between lasso and ridge
    best_linear_model = cancellation_cost_best_model(X, y, X_test, y_test)

    # loading data
    dataFrame: pd.DataFrame = load_data(path)

    # preproccess the data
    dataFrame = agoda_preprocess.unknown_data_preprocess(dataFrame)
    dataFrame = agoda_preprocess.preprocess_test(dataFrame, mValues)

    predictions = best_model.predict(dataFrame)

    # Filter the predictions to get the IDs with non-zero values
    non_zero_ids = predictions[predictions != 0].index

    # Make predictions using the best_linear_model for the non-zero IDs
    predicted_values = best_linear_model.predict(dataFrame.loc[non_zero_ids])

    # Create a DataFrame with the non-zero IDs and the predicted values
    output_df = pd.DataFrame({'id': dataFrame.loc[non_zero_ids]['h_booking_id'], 'prediction': predicted_values})

    # Assign -1 as the prediction for the IDs with prediction value of 0
    output_df.append(pd.DataFrame({'id': dataFrame.loc[predictions == 0]['h_booking_id'], 'prediction': -1}))

    # Save the DataFrame to a CSV file
    output_df = pd.DataFrame({'id': dataFrame['h_booking_id'], 'cancellation': predictions})
    output_df.to_csv('agoda_cost_of_cancellation.csv', index=False)
