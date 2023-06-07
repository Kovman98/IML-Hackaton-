import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
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
    df = pd.read_csv(filename, parse_dates=['checkin_date'])
    df = df[(df['Temp'] > -15) & (df['Year'] > 1990)].reset_index()
    # df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("/Users/EyalsMac/University/Year 2/Semester B/Introduction to Machine Learning/IML.HUJI-main/datasets/Mission1/agoda_cancellation_train.csv")


    # # Question 2 - Exploring data for specific country
    israel_dataset = X[X['Country'] == 'Israel'].reset_index()
    fig = px.scatter(israel_dataset, x='DayOfYear', y='Temp', color='Year', title='Average Daily Temp by Day Of Year')
    fig.show()
    israel_grouped = israel_dataset.groupby('Month').agg({'Temp': 'std'}).reset_index()
    fig = px.bar(israel_grouped, x='Month', y='Temp', title = 'Standard Deviation of Daily Temp by Month',
                 labels={'Temp': 'Standard Deviation', 'Month': 'Month'})
    fig.show()

    # # Question 3 - Exploring differences between countries
    countries_months_grouped = X.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    fig = px.line(countries_months_grouped, x='Month', y='mean', color ='Country', error_y='std',
                  title='Average Monthly Temp by Country',
                  labels={'mean': 'Average Temp', 'Month': 'Month', 'std': 'Standard Deviation'})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    (train_X, train_y, test_X, test_y) = split_train_test(israel_dataset['DayOfYear'], israel_dataset['Temp'])
    loss = np.zeros(10)
    for i in range(1, 11):
        polynomial_regression = PolynomialFitting(i).fit(train_X.to_numpy(), train_y.to_numpy())
        loss[i-1] = polynomial_regression.loss(test_X.to_numpy(), test_y.to_numpy())
        loss[i-1] = np.round(loss[i-1], 2)
        print(loss[i-1])
    degrees = list(range(1, 11))
    loss = pd.DataFrame(dict(k=degrees, loss=loss))
    fig = px.bar(loss, x="k", y="loss", title=" Test Error as a function of Polynomial Degree K")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    degree = 5
    loss_per_country = {}
    for name in X['Country'].unique():
        if name == 'Israel':
            continue
        polynomial_regression = PolynomialFitting(degree).fit(train_X.to_numpy(), train_y.to_numpy())
        test_country_x = X[X['Country'] == name]['DayOfYear'].reset_index()['DayOfYear']
        test_country_y = X[X['Country'] == name]['Temp'].reset_index()['Temp']
        loss_per_country[name] = polynomial_regression.loss(test_country_x.to_numpy(), test_country_y.to_numpy())
    df = pd.DataFrame.from_dict(loss_per_country, orient='index', columns=['Loss']).reset_index()
    df.rename(columns={'index': 'Country'}, inplace=True)
    fig = px.bar(df, x='Country', y='Loss', title='Temperature Loss by Country',
                 labels={'Country': 'Country', 'Loss': 'Loss'},color='Country')
    fig.show()
