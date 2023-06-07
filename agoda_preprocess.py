import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


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
    df = pd.read_csv(filename)

    return df



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("/Users/EyalsMac/University/Year 2/Semester B/Introduction to Machine Learning/IML.HUJI-main/datasets/Mission1/agoda_cancellation_train.csv")
