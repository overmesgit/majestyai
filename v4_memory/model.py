import torch
import torch.nn as nn
from torch import optim


class StateModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12, rate=0.01):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=rate)

    def predict(self, input_data):
        self.optimizer.zero_grad()
        return self(torch.tensor(input_data, dtype=torch.float32))

    def train_once(self, input_data, target_data):
        self.optimizer.zero_grad()
        predictions = self(torch.tensor(input_data, dtype=torch.float32))

        target = torch.tensor(target_data, dtype=torch.float32)
        total_loss = self.criterion(predictions, target)
        total_loss.backward()
        self.optimizer.step()
        return total_loss

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


class GenericModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, output_size)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def predict(self, input_data):
        self.optimizer.zero_grad()
        return self(torch.tensor(input_data, dtype=torch.float32))

    def train_once(self, input_data, target_data, lr=None):
        optimizer = self.optimizer
        optimizer.zero_grad()
        predictions = self(torch.tensor(input_data, dtype=torch.float32))

        target = torch.tensor(target_data, dtype=torch.float32)
        total_loss = self.criterion(predictions, target)
        total_loss.backward()
        optimizer.step()
        return total_loss

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
