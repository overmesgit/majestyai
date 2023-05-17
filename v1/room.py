import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data


# Define the neural network model
class RoomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = RoomClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def predict_rooms(rooms, char):
    # Character predicts outcome for each room
    room_inputs = torch.tensor([(room.level, char.level) for room in rooms], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(room_inputs)
    return room_inputs, predictions


def update_rooms(actual_win_prob, current_input):
    # Update training data
    targets = torch.tensor([actual_win_prob], dtype=torch.float32).view(-1, 1)
    inputs = current_input.unsqueeze(0)

    # Train the model with the new data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


def update_room_batch(input_targets: list):
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

        # Print the loss for this epoch
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # if abs(loss.item()) < 0.3:
        #     break

    print(f"Room Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
