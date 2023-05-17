import torch

from room import model

# New data (enemy_score, character_status)
new_data = [
    (15, 90),
    (25, 70),
    (35, 50),
]

# Convert the new data into a tensor
new_inputs = torch.tensor(new_data, dtype=torch.float32)

# Pass the new inputs through the trained model
with torch.no_grad():
    predictions = model(new_inputs)

# Convert the output predictions into a list of probabilities
predicted_probabilities = predictions.squeeze().tolist()

# Print the predicted probabilities
for i, (es, cs) in enumerate(new_data):
    print(
        f"Enemy score: {es}, Character status: {cs}, Predicted possibility of winning: {predicted_probabilities[i]:.4f}")
