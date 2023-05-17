import torch
from torch import nn as nn, optim as optim
from torch.utils import data


class AttentionClassifier(nn.Module):
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


model_prior = AttentionClassifier()
criterion_prior = nn.BCELoss()
optimizer_prior = optim.Adam(model_prior.parameters(), lr=0.01)

if __name__ == '__main__':

    # Example input data (enemy_score, character_status, possibility_of_winning)
    room_data = [
        [
            (1, 0, 0, 0, 1, 0, 0),
            (0, 1, 0, 0, 0.1, 0, 0),
            (0, 0, 1, 0, 0.5, 0, 0),
            (0, 0, 0, 1, 1, 0, 0),
        ]
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
