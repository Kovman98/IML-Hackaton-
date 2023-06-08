import pandas as pd
import agoda_preprocess


def load_data(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path, parse_dates=['booking_datetime', 'checkout_date'], dayfirst=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df['checkin_date'] = pd.to_datetime(df['checkin_date'], format='%d/%m/%Y %H:%M')
    return df


def task_1(model, path: str, mean_values: pd.Series) -> None:
    # loading data
    df: pd.DataFrame = load_data(path)

    # preproccess the data
    df = agoda_preprocess.unknown_data_preprocess(df)
    df = agoda_preprocess.preprocess_test(df, mean_values)

    predictions = model.predict(df)

    output_df = pd.DataFrame({'id': df['h_booking_id'], 'cancellation': predictions})
    output_df.to_csv("agoda_cancellation_prediction.csv", index=False)
