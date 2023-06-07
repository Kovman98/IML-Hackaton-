from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    invalid_indexes = (X['bedrooms'] <= 0) | (X['sqft_lot15'] <= 0) | (X['sqft_living'] <= 0) | \
                      (X['sqft_lot'] <= 0) | (X['sqft_basement'] < 0) | (X['floors'] < 0) | (X['yr_renovated'] < 0) \
                      | (X['bathrooms'] < 0) | (X['sqft_living15'] <= 0)
    X = X[~invalid_indexes]
    insufficient_bedrooms = X['bedrooms'] > 13
    insufficient_sqft_lot = X['sqft_lot'] > 120000
    X = X[~insufficient_bedrooms]
    X = X[~insufficient_sqft_lot]
    if not y.empty:
        y = y[~invalid_indexes]
        y = y[~insufficient_bedrooms]
        y = y[~insufficient_sqft_lot]
        invalid_price_index = y < 0
        y = y[~invalid_price_index]
        X = X[~invalid_price_index]
    X['Renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    X['decade'] = (X['yr_built'] / 10).astype(int)
    X = X.drop(['lat', 'long', 'zipcode', 'waterfront', 'date', 'id', 'yr_renovated', 'yr_built'], axis=1)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        correlation = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(x=X[feature], y=y, title=f'Correlation between {feature} values and responses (Correlation:'
                                                  f' {correlation:.2f})',
                         labels={'x': f"{feature}", 'y': 'Price in $'})
        pio.write_image(fig, output_path + f"/pearson_correlation_{feature}.png")

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    # raise NotImplementedError()
    (train_X, train_y, test_X, test_y) = split_train_test(df.drop('price', axis=1), df['price'])
    # Checking the correlation of Bedrooms on Prices
    # bedroom_prices = df[df['bedrooms'] == 4]
    # plt.hist(bedroom_prices['price'], bins=40)
    #
    # bedroom_prices = df[df['bedrooms'] == 3]
    # plt.hist(bedroom_prices['price'], bins=40)
    #
    # bedroom_prices = df[df['bedrooms'] == 2]
    # plt.hist(bedroom_prices['price'], bins=40)
    #
    # bedroom_prices = df[df['bedrooms'] == 1]
    # plt.hist(bedroom_prices['price'], bins=40)
    # plt.show()
    #
    # Checking the influence of Lat and Long on house prices
    # x_ = np.unique(df['lat'].tolist())[1:-1]
    # y_ = np.unique(df['long'].tolist())[:-2]
    #
    # values = np.array(df['price'])
    # go.Figure(go.Heatmap(x=x_,y=y_,z=values), layout = go.Layout(height=500,width = 500)).show()
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=df['yr_built'], y=df['price'], mode='markers'))
    # fig.show()

    # Question 2 - Preprocessing of housing prices dataset
    (train_X, train_y) = preprocess_data(train_X, train_y)
    (test_X, test_y) = preprocess_data(test_X, test_y)

    # Question 3 - Feature evaluation with respect to response
    # raise NotImplementedError()
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
    samples = list(range(10,101))
    fitted_results = np.zeros((len(samples),10))
    for i in range(10, 101):
        fraction = i/100
        for j in range(0, 10):
            sample_x = train_X.sample(frac=fraction)
            sample_y = train_y.loc[sample_x.index]
            fitted_results[i-10, j] = LinearRegression(include_intercept=True).fit(sample_x.values, sample_y.values).loss(test_X.values, test_y.values)
    mean, std = fitted_results.mean(axis=1),fitted_results.std(axis=1)
    fig = go.Figure([go.Scatter(x=samples, y=mean,mode="markers+lines",name="Mean",
                                line=dict(dash="dash"),marker =dict(color="green", opacity=.7)),
                     go.Scatter(x=samples, y=mean-2*std, fill=None, mode="lines", name="mean-2*std" , line=dict(color="lightgrey")),
                     go.Scatter(x=samples, y=mean+2*std, fill="tonexty", mode="lines", name="mean+2*std", line=dict(color="lightgrey"))],
                    layout=go.Layout(title="MSE as Function of Training Percentage",
                    xaxis=dict(title="Training sample percentage"),
                    yaxis =dict(title="MSE"),
                    showlegend=False))
    fig.show()