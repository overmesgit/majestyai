import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim


class StateModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, rate=0.01):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=rate)
        self.scaler = None

    def predict(self, input_data):
        self.optimizer.zero_grad()
        input_data = [input_data]
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        return self(torch.FloatTensor(input_data))[0]

    def train_once(self, input_data, target_data):
        self.scaler = StandardScaler().fit(input_data)
        X_train = self.scaler.transform(input_data)
        X_train_tensor = torch.FloatTensor(X_train)
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
    def __init__(self, input_size, hidden=64, lr=0.01):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scaler = None

    def predict(self, input_data):
        self.optimizer.zero_grad()
        input_data = [input_data]
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        return self(torch.FloatTensor(input_data))[0]

    def train_once(self, input_data, target_data):
        self.scaler = StandardScaler().fit(input_data)
        X_train = self.scaler.transform(input_data)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(target_data)

        loss = 0
        epochs = 100
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
