import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import sklearn
from forex_python.converter import CurrencyRates

def preprocess_train(X: pd.DataFrame) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X['is_cancelled'] = X['cancellation_datetime'].fillna(0)
    X['is_cancelled'].loc[X['is_cancelled'] != 0] = 1
    X = X.drop(['h_booking_id', 'hotel_live_date', 'h_costumer_id', 'costumer_nationality',
                'origin_counter_code', 'language'], axis=1)
    X = X.drop_duplicates()
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['number_of_nights'] = (X['checkout_date'] - X['checkin_date']).dt.days

    X = X[X["days_before_checkin" > -2]]
    X = X[X["number_of_nights"] > 0]




    df = pd.get_dummies(df, prefix='hotel_country_code_', columns=['hotel_country_code'])
    df = pd.get_dummies(df, prefix='accommadation_type_name_', columns=['accommadation_type_name'])
    df = pd.get_dummies(df, prefix='guest_nationality_country_name_', columns=['guest_nationality_country_name'])
    df = pd.get_dummies(df, prefix='original_payment_type_', columns=['original_payment_type'])
    df = pd.get_dummies(df, prefix='cancellation_policy_code_', columns=['cancellation_policy_code'])
    df = pd.get_dummies(df, prefix='hotel_city_code_', columns=['hotel_city_code'])

    X = X[X["hotel_star_rating"].between(0, 5)]
    X = X[X["no_of_adults"].isin(range(20))]
    X = X[X["no_of_children"].isin(range(20))]
    X = X[X["no_of_extra_bed"].isin(range(3))]
    X = X[X["no_of_room"].isin(range(10))]
    return X



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = pd.read_csv("dataset/agoda_cancellation_train.csv", parse_dates=['booking_datetime', 'checkin_date',
                                                                         'checkout_date', 'cancellation_datetime'])
    X = preprocess_train(X)

    count = X['is_cancelled'].value_counts()
    labels = count.index.tolist()
    values = count.values.tolist()
    colors = ['#FF7F0E', '#1F77B4']  # Orange for Cancelled, Blue for not cancelled
    fig = go.Figure(data=[go.Pie(labels=labels,values=values,showlegend=False)])
    fig.update_traces(marker=dict(colors=colors))
    fig.update_layout(title="Cancellation Pie Chart")
    # fig.show()
    y = X['is_cancelled']
    X = X.drop('is_cancelled', axis=1)


    # for feature in X:
    #     correlation = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
    #     fig = px.scatter(x=X[feature], y=y, title=f'Correlation between {feature} values and responses (Correlation:'
    #                                               f' {correlation:.2f})',
    #                      labels={'x': f"{feature}", 'y': 'Price in $'})
    #     pio.show()

    # y = X['cancellation_datetime']
    # y = y.fillna(0)
    # y.loc[y != 0] = 1
    #
    X = X.drop('cancellation_datetime', axis=1)
