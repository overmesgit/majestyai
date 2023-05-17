import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data


class RestClassifier(nn.Module):
    def __init__(self):
        super(RestClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def predict_action(winning_prob):
    input_data = torch.tensor([winning_prob], dtype=torch.float32).unsqueeze(0)
    output = classifier_act(input_data)
    _, decision = torch.max(output, dim=1)
    return input_data, decision


def update_action(input_data, label):
    # Update training data
    label = torch.tensor([label], dtype=torch.long)

    optimizer_act.zero_grad()
    output = classifier_act(input_data)
    loss = criterion_act(output, label)
    loss.backward()
    optimizer_act.step()


def update_prior_batch(input_targets: list):
    inputs = torch.stack([inp for inp, _ in input_targets])
    targets = torch.tensor([tar for _, tar in input_targets], dtype=torch.float32).view(-1, 1)
    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)

    num_epochs = 30
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer_act.zero_grad()
            outputs = classifier_act(inputs)
            loss = criterion_act(outputs, targets)
            loss.backward()
            optimizer_act.step()

    print(f"Prior Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Create the classifier and set up the optimizer and loss function
classifier_act = RestClassifier()
optimizer_act = optim.SGD(classifier_act.parameters(), lr=0.1)
criterion_act = nn.CrossEntropyLoss()
