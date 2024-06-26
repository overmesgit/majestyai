import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the CSV file into a DataFrame
data = pd.read_csv('d1694931256.588213.txt')

# Assume the last column is the target and the rest are features
X = data.iloc[:, :8].values  # Features
y = data.iloc[:, 8:].values  # Targets

# Step 2: Split the data into training and testing sets
X_train, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize the features (optional but often helpful)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test_raw)

# Step 3: Convert the data into PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        hidden = 64
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.fc(x)


# Initialize and train the network
model = SimpleNN(X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

pred = model(X_test_tensor)
for x, x_raw, y, p in zip(X_test_tensor, X_test_raw, y_test_tensor, pred):
    print("=")
    print(x_raw)
    print(x)
    print(y)
    print(p)


print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")