import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from room import optimizer, model, criterion

# Example input data (enemy_score, character_status, possibility_of_winning)
room_data = [
    (10, 100, 0.9),
    (20, 80, 0.7),
    (30, 60, 0.4),
    (40, 30, 0.1),
]

inputs = torch.tensor([(es, cs) for es, cs, _ in room_data], dtype=torch.float32)
targets = torch.tensor([pw for _, _, pw in room_data], dtype=torch.float32).view(-1, 1)

# Create a dataset and DataLoader
dataset = data.TensorDataset(inputs, targets)
dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
