import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sns as sns
from matplotlib import pyplot as plt


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
    X = load_data("dataset/agoda_cancellation_train.csv")

    y = X['cancellation_datetime']
    y = y.fillna(0)
    y.loc[y != 0] = 1

    X = X.drop('cancellation_datetime', axis=1)
    figsize = (6 * 6, 20)
    fig = plt.figure(figsize=figsize)
    for idx, col in enumerate(X.columns):
        ax = plt.subplot(4, 5, idx + 1)
        sns.kdeplot(
            data=X, hue='cancellation_datetime', fill=True,
            x=col, palette=['#9E3F00', 'red'], legend=False
        )

        ax.set_ylabel(''); ax.spines['top'].set_visible(False),
        ax.set_xlabel(''); ax.spines['right'].set_visible(False)
        ax.set_title(f'{col}', loc='right',
                     weight='bold', fontsize=20)

    fig.suptitle(f'Features vs Target\n\n\n', ha='center', fontweight='bold', fontsize=25)
    fig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=25, ncol=3)
    plt.tight_layout()
    plt.show()