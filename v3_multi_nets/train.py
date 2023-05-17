# Example input data (enemy_score, character_status, possibility_of_winning)

import torch

from v3_multi_nets.areas import Hero, Room, Rest
from v3_multi_nets.model import GenericModel

room_data = [
    # type, health, gold, hero health, hero level, hero gold
    [(Hero(100, 1, 1), Room(50, 1)), (Hero(75, 1.1, 2), Room(0, 0))],
    [(Hero(100, 1, 1), Room(80, 1)), (Hero(60, 1.1, 2), Room(0, 0))],
    [(Hero(100, 1, 1), Room(40, 1)), (Hero(80, 1.1, 2), Room(0, 0))],
    [(Hero(80, 1, 1), Room(60, 1)), (Hero(50, 1.1, 2), Room(0, 0))],
    [(Hero(80, 1, 1), Room(80, 1)), (Hero(0, 1, 1), Room(80, 0))],
    [(Hero(80, 1, 1), Room(40, 1)), (Hero(60, 1.1, 2), Room(0, 0))],
    [(Hero(60, 1, 1), Room(80, 1)), (Hero(0, 1, 1), Room(80, 0))],
    [(Hero(60, 1, 1), Room(60, 1)), (Hero(0, 1, 1), Room(60, 0))],
    [(Hero(60, 1, 1), Room(40, 1)), (Hero(40, 1.1, 2), Room(0, 0))],
    [(Hero(80, 1, 1), Rest(10)), (Hero(90, 1, 1), Rest(0, ))],
    [(Hero(60, 1, 1), Rest(50)), (Hero(85, 1, 1), Rest(0, ))],
    [(Hero(20, 1, 1), Rest(80)), (Hero(100, 1, 1), Rest(0, ))],
    [(Hero(100, 1, 1), Rest(80)), (Hero(100, 1, 1), Rest(0, ))],
    [(Hero(80, 1, 1), Rest(80)), (Hero(100, 1, 1), Rest(0, ))],

]

# Initialize the model, loss functions, and optimizer
room_model = GenericModel(3 + 2, 3 + 2)
rest_model = GenericModel(3 + 1, 3 + 1)

# data_input = [(torch.tensor(h, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)) for
#               (h, r), _
#               in room_data]
# data_target = [(torch.tensor(h, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)) for
#                _, (h, r)
#                in room_data]

# Create a dataset and DataLoader
# dataset = data.TensorDataset(inputs, targets)
# dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)

if __name__ == '__main__':
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, target) in enumerate(room_data):
            hero, room = inputs
            if isinstance(room, Room):
                model = room_model
            else:
                model = rest_model

            t_hero, t_room = target
            total_loss = model.train_once(hero + room, t_hero + t_room)

        # Print the loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

    for i, (inputs, target) in enumerate(room_data):
        hero, room = inputs
        if isinstance(room, Room):
            model = room_model
        else:
            model = rest_model

        with torch.no_grad():
            predictions = model(torch.tensor(hero + room, dtype=torch.float32))

        print('#' * 10)
        print(inputs)
        print(target)
        print(predictions)
