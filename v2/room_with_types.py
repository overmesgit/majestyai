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
        self.fc4 = nn.Linear(12, 3)

    def reset_func(self, layer):
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


model = RoomClassifierFull()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def predict_rooms_full_status(rooms_input):
    # Character predicts outcome for each room
    room_inputs = torch.tensor(rooms_input, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(room_inputs)
    return room_inputs, predictions


def update_room_full_status(current_input, final_status):
    # Update training data
    targets = torch.tensor(final_status, dtype=torch.float32).view(1, -1)
    inputs = current_input.unsqueeze(0)

    # Train the model with the new data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


def update_room_full_batch_status(input_targets: list):
    # model.apply(model.reset_func)

    inputs = torch.stack([inp for inp, _ in input_targets])
    targets = torch.tensor([tar for _, tar in input_targets], dtype=torch.float32)
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
