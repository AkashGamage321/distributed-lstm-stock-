import os
import pandas as pd


def load_stock_files(data_dir):
    files = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            files.append(os.path.join(data_dir, file))

    return files


def load_stock_dataframe(file_path):
    df = pd.read_csv(file_path)

    df = df.sort_values("Date")

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df = df.dropna()

    df = df[:1000]

    return df