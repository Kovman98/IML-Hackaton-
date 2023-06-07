import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils.utils import split_train_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    data = pd.read_csv(filename, parse_dates= ["Date"])
    data = data.dropna()
    data = data.drop_duplicates()
    data = data[(data['Temp'] > 0)]
    data['data'] = pd.to_datetime(data['Date'])
    data['day_of_year'] = data['data'].dt.dayofyear
    data = data.drop(['Day','Date', 'data'], axis=1)
    #todo: what to do with the month
    print(data)
    return data




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset

    data = load_data("../datasets/City_Temperature.csv")
    #todo - check if this part is correct


    # Question 2 - Exploring data for specific country
    plt.figure()
    data_israel = data[data['Country'] == 'Israel']
    for year in data_israel['Year'].unique():
        specific_year = data_israel[data_israel['Year'] == year]
        # plt.scatter(specific_year['day_of_year'], specific_year['Temp'], label=year)

    # add labels and legend to the plot
    # plt.xlabel('Day of the Year')
    # plt.ylabel('Temperature')
    # plt.legend()
    # plt.show()


    month_list = np.zeros(12)
    std_list = np.zeros(12)
    for count,month in enumerate(data_israel['Month'].unique()):
        specific_month = data_israel[data_israel['Month'] == month]
        month_list[count] = month
        std_list[count] = specific_month['Temp'].std()

    # plt.bar(month_list, std_list)
    # plt.xlabel('months')
    # plt.ylabel('std of temperature')
    # plt.title('Bar Plot of std_temperature according to the month')
    # plt.show()


    # Question 3 - Exploring differences between countries
    data_group = data.groupby(['Country', 'Month']).agg(mean=("Temp", "mean"), std=("Temp", "std"))
    data_group = data_group.reset_index()
    # fig_1 = px.line(data_group, x='Month', y='mean', error_y='std',
    #               color='Country', title='Mean monthly temp according to country Country')
    # fig_1.show()

    # Question 4 - Fitting model for different values of `k`
    #todo - when did they split when did they preprocessed
    loss_list = np.zeros(10)
    k_list = np.zeros(10)
    x_train, y_train, x_test, y_test = split_train_test(data_israel['day_of_year'],data_israel['Temp'])
    x_train_array = x_train.values.flatten()
    y_train_array = y_train.values.flatten()
    x_test_array = x_test.values.flatten()
    y_test_array = y_test.values.flatten()
    for count,k in enumerate(range(1,11)):
        poly_model = PolynomialFitting(k)
        poly_model.fit(x_train_array,y_train_array)
        loss_list[count] = np.round(poly_model.loss(x_test_array, y_test_array),2)
        k_list[count] = k

    print(loss_list)
    plt.bar(k_list, loss_list)
    plt.xlabel('k values')
    plt.ylabel('loss of model')
    plt.title('test error recorded for each value of k')
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(5)
    poly_model.fit(x_train_array, y_train_array)
    loss_list= np.zeros(3)
    country_list = []
    count = 0
    for country in data['Country'].unique():
        if country != 'Israel':
            specific_country = data[data['Country'] == country]
            specificX_test_array = specific_country['day_of_year'].values.flatten()
            specificY_test_array = specific_country['Temp'].values.flatten()
            loss_list[count] = np.round(poly_model.loss(specificX_test_array, specificY_test_array), 2)
            country_list.append(country)
            count+=1

    plt.bar(country_list, loss_list)
    plt.xlabel('country')
    plt.ylabel('loss')
    plt.title('model’s error over each of the other countries')
    # plt.show()
