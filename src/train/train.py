import sys
sys.path.append("src")

import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import load_stock_files, load_stock_dataframe
from data.preprocess import create_sequences, normalize_data
from model.lstm import LSTMModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SEQ_LEN = 50
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

DATA_DIR = "data/stock/stocks"


def load_data():

    files = load_stock_files(DATA_DIR)

    df = load_stock_dataframe(files[0])

    data = df.values

    data, scaler = normalize_data(data)

    X, y = create_sequences(data, SEQ_LEN)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def train():

    X, y = load_data()

    model = LSTMModel().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    dataset = torch.utils.data.TensorDataset(X, y)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for epoch in range(EPOCHS):

        total_loss = 0

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss {total_loss:.4f}")


if __name__ == "__main__":
    train()