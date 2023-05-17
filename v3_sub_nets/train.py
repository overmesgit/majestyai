# Example input data (enemy_score, character_status, possibility_of_winning)
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils import data

from v3_sub_nets.model import MultiTaskModel

room_data = [
    # type, health, gold, hero health, hero level, hero gold
    [((100, 1, 1), (1, 50, 1)), ((75, 1.1, 2), (0, 0))],
    [((100, 1, 1), (1, 80, 1)), ((60, 1.1, 2), (0, 0))],
    [((100, 1, 1), (1, 40, 1)), ((80, 1.1, 2), (0, 0))],
    [((80, 1, 1), (1, 60, 1)), ((50, 1.1, 2), (0, 0))],
    [((80, 1, 1), (1, 80, 1)), ((0, 1, 1), (80, 0))],
    [((80, 1, 1), (1, 40, 1)), ((60, 1.1, 2), (0, 0))],
    [((60, 1, 1), (1, 80, 1)), ((0, 1, 1), (80, 0))],
    [((60, 1, 1), (1, 60, 1)), ((0, 1, 1), (60, 0))],
    [((60, 1, 1), (1, 40, 1)), ((40, 1.1, 2), (0, 0))],
    [((80, 1, 1), (2, 10)), ((90, 1, 1), (0,))],
    [((60, 1, 1), (2, 50)), ((85, 1, 1), (0,))],
    [((20, 1, 1), (2, 80)), ((100, 1, 1), (0,))],
    [((100, 1, 1), (2, 80)), ((100, 1, 1), (0,))],
    [((80, 1, 1), (2, 80)), ((100, 1, 1), (0,))],

]

# Initialize the model, loss functions, and optimizer
model = MultiTaskModel()
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data_input = [(torch.tensor(h, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)) for
              (h, r), _
              in room_data]
data_target = [(torch.tensor(h, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)) for
               _, (h, r)
               in room_data]

# Create a dataset and DataLoader
# dataset = data.TensorDataset(inputs, targets)
# dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(zip(data_input, data_target)):
        hero, room = targets
        optimizer.zero_grad()

        # Forward pass
        task1_predictions, task2_predictions, task3_predictions = model(inputs)

        # Compute losses for both tasks
        total_loss = criterion1(task1_predictions, hero)
        if room[0] == 1:
            loss2 = criterion2(task2_predictions, targets[1])
            total_loss += loss2
        elif room[0] == 2:
            loss3 = criterion3(task3_predictions, targets[1])
            total_loss += loss3

        # Combine losses and perform backpropagation
        total_loss.backward()
        optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

for (inputs, targets) in zip(data_input, data_target):
    with torch.no_grad():
        outputs = model(inputs)
    print('#' * 10)
    print(inputs)
    print(targets)
    print(outputs)
