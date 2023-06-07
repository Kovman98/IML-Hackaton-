import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
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
    X = X.drop(['h_booking_id'], axis=1)
    X = X.drop_duplicates()
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days

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
