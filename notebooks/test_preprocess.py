import sys
sys.path.append("src")

from data.loader import load_stock_files, load_stock_dataframe
from data.preprocess import create_sequences, normalize_data


data_dir = "data/stock/stocks"

files = load_stock_files(data_dir)

print("files:", len(files))

df = load_stock_dataframe(files[0])

print(df.head())


data = df.values

data, scaler = normalize_data(data)

X, y = create_sequences(data, 50)

print("X shape:", X.shape)
print("y shape:", y.shape)