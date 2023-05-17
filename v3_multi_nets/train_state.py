import torch

from v3_multi_nets.model import GenericModel
from v3_multi_nets.areas import Hero

room_data = [
    [Hero(100, 1, 1), Hero(-20, 0.1, 1), 0.4],
    [Hero(100, 1, 1), Hero(-50, 0.1, 1), 0.3],
    [Hero(100, 1, 1), Hero(-70, 0.1, 1), 0.2],
    [Hero(80, 1, 1), Hero(-40, 0.1, 1), 0.2],
    [Hero(80, 1, 1), Hero(-60, 0.1, 1), 0.1],
    [Hero(80, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-40, 0.1, 1), 0.1],
    [Hero(60, 1, 1), Hero(-60, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    # heal
    [Hero(80, 1, 1), Hero(20, 0, 0), 0.3],
    [Hero(60, 1, 1), Hero(60, 0, 0), 0.35],
    [Hero(20, 1, 1), Hero(80, 0, 0), 0.4],
    [Hero(100, 1, 1), Hero(0, 0, 0), 0],
    [Hero(80, 1, 1), Hero(20, 0, 0), 0.1],

]

state_model = GenericModel(6, 1)


def pre_train_state_model():
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (init_state, diff_state, target) in enumerate(room_data):
            total_loss = state_model.train_once(init_state + diff_state, [target])

        # Print the loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

    torch.save(state_model.state_dict(), 'state_model.pth')

    for i, (init_state, diff_state, target) in enumerate(room_data):

        with torch.no_grad():
            predictions = state_model(torch.tensor(init_state + diff_state, dtype=torch.float32))

        print('#' * 10)
        print(init_state + diff_state)
        print(target)
        print(predictions)

if __name__ == '__main__':
    pre_train_state_model()