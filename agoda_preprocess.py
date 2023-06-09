import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
import sklearn
import requests
import datetime
import re


class MeanValuesCalculator:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.mean_values = self.calculate_mean_values()

    def calculate_mean_values(self):
        mean_values = self.dataframe.mean()
        return mean_values


# def split_policy(policies):
#     # Regular expression pattern
#     pattern = r"(\d+)D(\d+)([PN])"
#
#     # Split the policies
#     policies = policies.split('_')
#
#     # Apply regex to each policy
#     results = []
#     for policy in policies:
#         match = re.match(pattern, policy)
#         if match:
#             days, charge, charge_type = match.groups()
#
#             # Convert to proper types
#             days = int(days)
#             charge = int(charge)
#             charge_type = 'Percentage' if charge_type == 'P' else 'Nights'
#
#             results.append((days, charge, charge_type))
#
#     return results
#
#
# def apply_policy(days_diff, cancelled, policy_str):
#     policies = split_policy(policy_str)
#
#     # Sort policies by days, in descending order
#     policies.sort(key=lambda x: x[0], reverse=True)
#
#     for policy in policies:
#         policy_days, charge, charge_type = policy
#         if days_diff <= policy_days:
#             if cancelled:
#                 if charge_type == 'P' and charge > 0:  # They would pay a charge
#                     return -1
#                 else:  # They wouldn't pay a charge (charge is 0 or it's charged per nights, which is not applicable here)
#                     return 1
#             else:  # Didn't cancel
#                 return 0
#
#     # If no policy applies and they cancelled, they wouldn't pay a charge
#     if cancelled:
#         return 1
#
#     # If no policy applies and they didn't cancel
#     return 0

def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    # dropping unneeded columns
    X = X.drop(['hotel_live_date', 'h_customer_id', 'customer_nationality', 'origin_country_code', 'language',
                'original_payment_currency', 'original_payment_method', 'hotel_brand_code', 'hotel_id',
                'hotel_chain_code'], axis=1)

    # creating new columns for needed info
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['number_of_nights'] = (X['checkout_date'] - X['checkin_date']).dt.days

    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['number_of_nights'] = (X['checkout_date'] - X['checkin_date']).dt.days

    X['did_request'] = X['request_highfloor']+X['request_largebed']+X['request_nonesmoke']+X['request_latecheckin']\
                       +X['request_largebed']+X['request_twinbeds']+X['request_airport']+X['request_earlycheckin']
    X['did_request'] = X['did_request'].fillna(0)
    X['did_request'] = X['did_request'].apply(lambda x: 1 if x != 0 else 0)

    # preprocessing the cancellation policy
    X['cancellation_policy_code'] = X['cancellation_policy_code'].fillna('')
    X['cancel_for_free'] = 0  # default value

    for i, cancellation_policy in enumerate(X['cancellation_policy_code']):
        # need to check if inside policy
        if cancellation_policy != '':
            cancellation_parts = cancellation_policy.split('_')
            for part in cancellation_parts:
                if 'D' in part:
                    first, second = part.split('D')[0], part.split('D')[1]
                    if int(first) <= X['days_before_checkin'].loc[i] and second[0] != '0':
                        X['cancel_for_free'].loc[i] = 1

    # dropping columns that not needed (created alternative columns for these ones)
    X = X.drop(['checkin_date', 'checkout_date', 'cancellation_policy_code', 'request_largebed',
                'request_twinbeds', 'request_airport', 'request_earlycheckin', 'request_nonesmoke',
                'request_latecheckin', 'request_highfloor', 'booking_datetime'], axis=1)

    X['is_user_logged_in'] = X['is_user_logged_in'].astype(bool).astype(int)
    X['is_first_booking'] = X['is_first_booking'].astype(bool).astype(int)

    # using get_dummies on values labels that their values has no numerical meaning
    # X = pd.get_dummies(X, prefix='accommadation_type_name_', columns=['accommadation_type_name'], dtype=int)
    # X = pd.get_dummies(X, prefix='charge_option_', columns=['charge_option'], dtype=int)
    # X = pd.get_dummies(X, prefix='original_payment_type_', columns=['original_payment_type'], dtype=int)

    X = X.drop(['charge_option','original_payment_type','accommadation_type_name', 'hotel_country_code', 'guest_nationality_country_name', 'hotel_city_code'], axis=1)

    return X


def preprocess_test(X: pd.DataFrame, mValues: pd.Series):
    X = X.replace("nan", np.nan)

    # Iterate over each column in the test data
    for col in X.columns:
        # Check if the column has missing values
        if X[col].isnull().any():
            # Replace missing values with the mean value from mValues
            X[col] = X[col].fillna(mValues[col])

    return X


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

    X = X.drop_duplicates()

    X = X[X["number_of_nights"] > 0]
    X = X[X["days_before_checkin"] > -2]
    X = X[X["hotel_star_rating"].between(0, 5)]
    X = X[X["no_of_adults"].isin(range(12))]
    X = X[X["no_of_children"].isin(range(6))]
    X = X[X["no_of_extra_bed"].isin(range(3))]
    X = X[X["no_of_room"].isin(range(10))]
    X = X[X["original_selling_amount"] <= 5000]
    X = X[X['number_of_nights'].isin(range(10))]
    X = X[X['days_before_checkin'] <= 200]

    X = X.drop(['cancellation_datetime'], axis=1)
    X.reset_index()
    return X


def make_distribution_graphs(df: pd.DataFrame) -> None:

    # Filter data for cancelled and uncancelled bookings
    cancelled_bookings = df[df['is_cancelled'] == 1]
    uncancelled_bookings = df[df['is_cancelled'] == 0]

    # Create figure and subplots
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=['Distribution of Number of Nights', 'Distribution of Days Before Check-in',
                                        'Distribution of Hotel Star Rating', 'Distribution of Number of Adults',
                                        'Distribution of Number of Children', 'Distribution of Number of Rooms',
                                        'Distribution of Original Selling Amount', 'Distribution of First Booking'])

    # Define the parameter names
    parameters = ['number_of_nights', 'days_before_checkin', 'hotel_star_rating', 'no_of_adults', 'no_of_children',
                  'no_of_room',
                  'original_selling_amount', 'is_first_booking']

    # Add traces for each parameter
    for i, parameter in enumerate(parameters):
        fig.add_trace(go.Histogram(x=cancelled_bookings[parameter], name='Cancelled', opacity=0.65),
                      row=i // 4 + 1, col=i % 4 + 1)
        fig.add_trace(go.Histogram(x=uncancelled_bookings[parameter], name='Uncancelled', opacity=0.65),
                      row=i // 4 + 1, col=i % 4 + 1)

    # Update layout
    fig.update_layout(barmode='overlay', title='Distribution of Parameters for Cancelled and Uncancelled Bookings',
                      showlegend=False)

    # Update subplot titles
    titles = ['Distribution of Number of Nights', 'Distribution of Days Before Check-in',
              'Distribution of Hotel Star Rating',
              'Distribution of Number of Adults', 'Distribution of Number of Children',
              'Distribution of Number of Rooms',
              'Distribution of Original Selling Amount', 'Distribution of First Booking']

    fig.update_layout(barmode='overlay', title='Distribution of Parameters for Cancelled and Uncancelled Bookings',
                      showlegend=False)

    # Add axis titles
    titles = ['Number of Nights', 'Days Before Check-in', 'Hotel Star Rating', 'Number of Adults',
              'Number of Children', 'Number of Rooms', 'Original Selling Amount', 'First Booking']

    for i, title in enumerate(titles):
        fig.update_xaxes(title_text=title, row=i // 4 + 1, col=i % 4 + 1)

    # Show the plot
    fig.show()
