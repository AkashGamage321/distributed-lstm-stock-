import sys
sys.path.append("src")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.loader import load_stock_files, load_stock_dataframe
from data.preprocess import create_sequences, normalize_data
from model.lstm import LSTMModel


SEQ_LEN = 50
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

DATA_DIR = "data/stock/stocks"


def setup():
    dist.init_process_group(backend="nccl")


def cleanup():
    dist.destroy_process_group()


def load_data():

    files = load_stock_files(DATA_DIR)

    df = load_stock_dataframe(files[0])

    data = df.values

    data, scaler = normalize_data(data)

    X, y = create_sequences(data, SEQ_LEN)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def main():

    setup()

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    X, y = load_data()

    dataset = torch.utils.data.TensorDataset(X, y)

    sampler = DistributedSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    model = LSTMModel().to(device)

    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        sampler.set_epoch(epoch)

        total_loss = 0

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch} Loss {total_loss:.4f}")

    cleanup()


if __name__ == "__main__":
    main()