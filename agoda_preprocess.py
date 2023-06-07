import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import sklearn
import requests
import datetime
import re

import re

def split_policy(policies):
    # Regular expression pattern
    pattern = r"(\d+)D(\d+)([PN])"

    # Split the policies
    policies = policies.split('_')

    # Apply regex to each policy
    results = []
    for policy in policies:
        match = re.match(pattern, policy)
        if match:
            days, charge, charge_type = match.groups()

            # Convert to proper types
            days = int(days)
            charge = int(charge)
            charge_type = 'Percentage' if charge_type == 'P' else 'Nights'

            results.append((days, charge, charge_type))

    return results


def apply_policy(days_diff, cancelled, policy_str):
    policies = split_policy(policy_str)

    # Sort policies by days, in descending order
    policies.sort(key=lambda x: x[0], reverse=True)

    for policy in policies:
        policy_days, charge, charge_type = policy
        if days_diff <= policy_days:
            if cancelled:
                if charge_type == 'P' and charge > 0:  # They would pay a charge
                    return -1
                else:  # They wouldn't pay a charge (charge is 0 or it's charged per nights, which is not applicable here)
                    return 1
            else:  # Didn't cancel
                return 0

    # If no policy applies and they cancelled, they wouldn't pay a charge
    if cancelled:
        return 1

    # If no policy applies and they didn't cancel
    return 0


# Then, you can use this function to create a new column in your DataFrame:


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
    X = X.drop(['h_booking_id', 'hotel_live_date', 'h_customer_id', 'customer_nationality',
                'origin_country_code', 'language','original_payment_currency'], axis=1)
    X = X.drop_duplicates()
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['number_of_nights'] = (X['checkout_date'] - X['checkin_date']).dt.days
    X['cancellation_datetime'].fillna(-1)
    for i in range(len(X[:0])):
        if (X['cancellation_datetime'][i] != -1):
            X['days_difference'][i] = (X['cancellation_datetime'][i] - X['checkin_date'][i]).dt.days

    X = X[X["days_before_checkin"] > -2]
    X = X[X["number_of_nights"] > 0]



    # conversion_rates = {
    #     'AED': 0.272,'ARS': 0.004,'AUD': 0.665,'BDT': 0.0092,'BHD': 2.659,'BRL': 0.202,'CAD': 0.747,
    #     'CHF': 1.098,'CNY': 0.1402,'CZK': 0.452,'DKK': 0.1436,'EGP': 0.0323,'EUR': 1.0699,
    #     'FJD': 0.4471,'GBP': 1.2438,'HKD': 0.1275,'HUF': 0.0029,'IDR': 0.0000671,'ILS': 0.2735,'INR': 0.0121,
    #     'JOD': 1.41,'JPY': 0.00713,'KHR': 0.0002421,'KRW': 0.000764,'KWD': 3.249,'KZT': 0.00224,'LAK': 0.0000554,
    #     'LKR': 0.003427,'MXN': 0.0575,'MYR': 0.2173,'NGN': 0.00216,'NOK': 0.0906,'NZD': 0.6035,'OMR': 2.5973,
    #     'PHP': 0.0178,'PKR': 0.00348,'PLN': 0.2384,'QAR': 0.274,'RON': 0.2157,'RUB': 0.0122,'SAR': 0.2666,
    #     'SEK': 0.09197,'SGD': 0.7415,'THB': 0.028705,'TRY': 0.04302,'TWD': 0.0324,'UAH': 0.02711,'USD': 1,
    #     'VND': 0.0000427,'XPF': 0.00896,'ZAR': 0.052367
    # }


    # def convert_to_USD(row):
    #     amount = row['original_selling_amount']
    #     currency = row['original_payment_currency']
    #     if currency in conversion_rates:
    #         return amount*conversion_rates[currency]
    #     else:
    #         return amount

    # X["payment_nis"] = X.apply(convert_to_USD,axis = 1)

    X = pd.get_dummies(X, prefix='hotel_country_code_', columns=['hotel_country_code'])
    X = pd.get_dummies(X, prefix='accommadation_type_name_', columns=['accommadation_type_name'])
    X = pd.get_dummies(X, prefix='guest_nationality_country_name_', columns=['guest_nationality_country_name'])
    X = pd.get_dummies(X, prefix='original_payment_type_', columns=['original_payment_type'])
    #X = pd.get_dummies(X, prefix='cancellation_policy_code_', columns=['cancellation_policy_code'])
    X = pd.get_dummies(X, prefix='hotel_city_code_', columns=['hotel_city_code'])


    X['Cancellation Policy Applied'] = X.apply(
        lambda row: apply_policy(row['days_difference'], row['Cancelled'], row['Policy']), axis=1)

    X = X[X["hotel_star_rating"].between(0, 5)]
    X = X[X["no_of_adults"].isin(range(12))]
    X = X[X["no_of_children"].isin(range(6))]
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
