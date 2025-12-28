import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gru_model import TrafficGRU


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_and_save(csv_path, save_path='data/gru_weights.pth'):
    print(f"[*] Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)

    data = df['Bytes'].values.astype(float)
    max_val = np.max(data)
    data_normalized = data / max_val

    SEQ_LENGTH = 10
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 0.001

    X, y = create_sequences(data_normalized, SEQ_LENGTH)

    if len(X) == 0:
        print("[!] Not enough data to train. Run traffic generator longer.")
        return

    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)  # (Batch, Seq, Feature)
    y_tensor = torch.from_numpy(y).float().unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TrafficGRU(input_dim=1, hidden_dim=32, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[*] Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"[*] Model weights saved to {save_path}")

    with open('data/norm_factor.txt', 'w') as f:
        f.write(str(max_val))


if __name__ == '__main__':
    train_and_save('data/traffic_series.csv')