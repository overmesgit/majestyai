import torch
import torch.nn as nn
from torch import optim


class StateModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, rate=0.01):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=rate)

    def predict(self, input_data):
        self.optimizer.zero_grad()
        return self(torch.tensor(input_data, dtype=torch.float32))

    def train_once(self, input_data, target_data):
        X_train_tensor = torch.FloatTensor(input_data)
        y_train_tensor = torch.FloatTensor(target_data)

        loss = 0
        epochs = 50
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        return loss

    def forward(self, x):
        return self.fc(x)


class GenericModel(nn.Module):
    def __init__(self, input_size, hidden=8, lr=0.001):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def predict(self, input_data):
        self.optimizer.zero_grad()
        return self(torch.tensor(input_data, dtype=torch.float32))

    def train_once(self, input_data, target_data):
        X_train_tensor = torch.FloatTensor(input_data)
        y_train_tensor = torch.FloatTensor(target_data)

        loss = 0
        epochs = 50
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        return loss

    def forward(self, x):
        return self.fc(x)
