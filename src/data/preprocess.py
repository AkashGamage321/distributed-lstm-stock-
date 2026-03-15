import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_length=50):

    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def normalize_data(data):

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler