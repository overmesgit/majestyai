import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data


# Define the neural network model
class RoomClassifierFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(12, 1)

    def reset_func(self, layer):
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


model = RoomClassifierFull()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def predict_rooms_full(rooms_input):
    # Character predicts outcome for each room
    room_inputs = torch.tensor(rooms_input, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(room_inputs)
    return room_inputs, predictions


def update_room_full(current_input, actual_win_prob):
    # Update training data
    targets = torch.tensor([actual_win_prob], dtype=torch.float32).view(-1, 1)
    inputs = current_input.unsqueeze(0)

    # Train the model with the new data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


def update_room_full_batch(input_targets: list):
    # model.apply(model.reset_func)

    inputs = torch.stack([inp for inp, _ in input_targets])
    targets = torch.tensor([tar for _, tar in input_targets], dtype=torch.float32).view(-1, 1)
    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    print(f"Room Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == '__main__':

    # Example input data (enemy_score, character_status, possibility_of_winning)
    room_data = [
        (1, 0, 50, 1, 10, 0.1),
        (1, 0, 50, 1, 100, 0.8),
        (1, 0, 30, 1, 30, 0.5),
        (1, 0, 60, 1, 50, 0.4),
        (1, 0, 10, 1, 50, 0.9),
        (0, 1, 0, 0, 50, 0),
        (0, 1, 50, 0, 100, 0),
        (0, 1, 100, 0, 100, 0),
        (0, 1, 100, 0, 0, 0.5),
        (0, 1, 50, 0, 50, 0.25),
        (0, 1, 100, 0, 80, 0.1),
    ]

    inputs = torch.tensor([d for *d, _ in room_data], dtype=torch.float32)
    targets = torch.tensor([pw for *_, pw in room_data], dtype=torch.float32).view(-1, 1)

    # Create a dataset and DataLoader
    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Print the loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    inputs = torch.tensor([d for *d, _ in room_data], dtype=torch.float32)
    targets = torch.tensor([pw for *_, pw in room_data], dtype=torch.float32).view(-1, 1)
    for (inputs, targets) in zip(inputs, targets):
        with torch.no_grad():
            outputs = model(inputs)
        print(inputs, targets, outputs)
