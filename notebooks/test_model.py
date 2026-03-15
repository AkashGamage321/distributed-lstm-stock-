import sys
sys.path.append("src")

import torch
from model.lstm import LSTMModel


model = LSTMModel()

print(model)


x = torch.randn(32, 50, 5)

y = model(x)

print("output shape:", y.shape)