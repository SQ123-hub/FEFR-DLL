import torch
import torch.nn as nn
import numpy as np


class TrafficGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_layers=1):
        super(TrafficGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn) = self.gru(x, h0.detach())

        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_data, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        inputs, labels = train_data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model


def predict_next_step(model, history_seq):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(history_seq).float().view(1, -1, 1)  # Batch=1
        prediction = model(x)
    return prediction.item()