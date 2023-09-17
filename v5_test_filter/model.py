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
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def predict(self, input_data):
        self.optimizer.zero_grad()
        return self(torch.tensor(input_data, dtype=torch.float32))

    def train_once(self, input_data, target_data):
        self.optimizer.zero_grad()
        predictions = self(torch.tensor(input_data, dtype=torch.float32))

        target = torch.tensor(target_data, dtype=torch.float32)
        print('predicted', predictions)
        print('target', target)
        total_loss = self.criterion(predictions, target)
        total_loss.backward()
        self.optimizer.step()
        return total_loss

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ParamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

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
        return x


class WeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        with torch.no_grad():
            self.fc.weight.fill_(1)
            self.fc.bias.fill_(0)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def train_once(self, input_data, target_data):
        self.optimizer.zero_grad()
        predictions = self(torch.tensor(input_data, dtype=torch.float32))

        target = torch.tensor(target_data, dtype=torch.float32)
        total_loss = self.criterion(predictions, target)
        total_loss.backward()
        self.optimizer.step()
        return total_loss

    def forward(self, x):
        return self.fc(x)


params_count = 8


class CombinedNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.param_nets = [[None] * params_count] * params_count
        for i in range(params_count):
            for j in range(params_count):
                self.param_nets[i][j] = ParamNet()

        self.weight_nets = [None] * 8
        for i in range(params_count):
            self.weight_nets[i] = WeightNet()

    def predict(self, input_data):
        with torch.no_grad():
            return self(torch.tensor(input_data, dtype=torch.float32))

    def train_once(self, input_data, target_data):

        input_ten = torch.tensor(input_data, dtype=torch.float32)
        target_ten = torch.tensor(target_data, dtype=torch.float32)
        x1, y1 = torch.meshgrid(input_ten, input_ten)
        paired_tensor = torch.stack((x1, y1), dim=-1)
        for i in range(params_count):
            pred_params = [None] * params_count
            for j in range(params_count):
                self.param_nets[i][j].optimizer.zero_grad()
                predictions = self.param_nets[i][j](paired_tensor[i][j])
                pred_params[j] = predictions
                total_loss = self.param_nets[i][j].criterion(predictions, target_ten[i])
                total_loss.backward()
                self.param_nets[i][j].optimizer.step()

            self.weight_nets[i].optimizer.zero_grad()
            pred_params_tensor = torch.tensor(pred_params, dtype=torch.float32)
            weight_target = 1 - (pred_params_tensor - target_ten[i]).abs()

            weight_pred = self.weight_nets[i](input_ten)
            total_loss = self.weight_nets[i].criterion(weight_pred, weight_target)
            total_loss.backward()
            self.weight_nets[i].optimizer.step()
        return 0

    def forward(self, x):
        params_output = [[None] * params_count] * params_count
        weight_output = [None] * params_count
        res = []

        x1, y1 = torch.meshgrid(x, x)
        paired_tensor = torch.stack((x1, y1), dim=-1)
        for i in range(params_count):
            for j in range(params_count):
                params_output[i][j] = self.param_nets[i][j](paired_tensor[i][j])
            w = self.weight_nets[i](x)
            weight_output[i] = w

            normalized_tensor = w / w.sum()
            final_output = (torch.stack(params_output[i]).squeeze() * normalized_tensor).sum()
            res.append(final_output)

        return params_output, weight_output, res
