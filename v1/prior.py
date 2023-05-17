import torch
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils import data


class PriorityClassifier(nn.Module):
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


model_prior = PriorityClassifier()
criterion_prior = nn.BCELoss()
optimizer_prior = optim.Adam(model_prior.parameters(), lr=0.01)


def predict_prior(rooms, predictions):
    prior_list = [(room.gold, pred) for (room, pred) in zip(rooms, predictions)]
    prior_inputs = torch.tensor(prior_list, dtype=torch.float32)
    with torch.no_grad():
        prior = model_prior(prior_inputs)
    return prior_inputs, prior


def update_prior_batch(input_targets: list):
    inputs = torch.stack([inp for inp, _ in input_targets])
    targets = torch.tensor([tar for _, tar in input_targets], dtype=torch.float32).view(-1, 1)
    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)

    # scheduler = ExponentialLR(optimizer_prior, gamma=0.8)
    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer_prior.zero_grad()
            outputs = model_prior(inputs)
            loss = criterion_prior(outputs, targets)
            loss.backward()
            optimizer_prior.step()

        # Print the loss for this epoch
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # if abs(loss.item()) < 0.3:
        #     break

        # scheduler.step()
    print(f"Prior Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def update_prior(prior_prob, current_input):
    targets = torch.tensor([prior_prob], dtype=torch.float32).view(-1, 1)
    inputs = current_input.unsqueeze(0)

    # Train the model with the new data
    optimizer_prior.zero_grad()
    outputs = model_prior(inputs)
    loss = criterion_prior(outputs, targets)
    loss.backward()
    optimizer_prior.step()


if __name__ == '__main__':

    # Example input data (enemy_score, character_status, possibility_of_winning)
    room_data = [
        (5, 1, 1),
        (4, 1, 0.8),
        (3, 1, 0.6),
        (2, 1, 0.4),
        (1, 1, 0.2),
        (2, 0.8, 0.2),
        (3, 0.6, 0.2),
        (4, 0.3, 0.1),
        (4, 0.2, 0.1),
        (4, 0.1, 0.1),
        (4, 0, 0),
        (5, 0, 0),
    ]

    inputs = torch.tensor([(es, cs) for es, cs, _ in room_data], dtype=torch.float32)
    targets = torch.tensor([pw for _, _, pw in room_data], dtype=torch.float32).view(-1, 1)
    print(inputs)
    print(targets)

    # Create a dataset and DataLoader
    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer_prior.zero_grad()
            outputs = model_prior(inputs)
            loss = criterion_prior(outputs, targets)
            loss.backward()
            optimizer_prior.step()

        # Print the loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    inputs = torch.tensor([(es, cs) for es, cs, _ in room_data], dtype=torch.float32)
    targets = torch.tensor([pw for _, _, pw in room_data], dtype=torch.float32).view(-1, 1)
    for (inputs, targets) in zip(inputs, targets):
        with torch.no_grad():
            outputs = model_prior(inputs)
        print(inputs, targets, outputs)
