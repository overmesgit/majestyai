import torch
import torch.nn as nn
import torch.optim as optim


class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()

        dropout_rate = 0.3
        shared_hidden_size = 12

        hidden_size = 6

        # My status intput
        hero_input = 3
        self.hero_input = nn.Sequential(
            nn.Linear(hero_input, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        # Task 1 layers
        room_input = 2
        self.room_input = nn.Sequential(
            nn.Linear(room_input, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        # Task 2 layers
        rest_input = 1
        self.rest_input = nn.Sequential(
            nn.Linear(rest_input, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_size * 3, shared_hidden_size),
            nn.Linear(shared_hidden_size, shared_hidden_size),
            nn.Linear(shared_hidden_size, shared_hidden_size),
        )

        # My status output
        self.hero_output = nn.Sequential(
            nn.Linear(shared_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hero_input),
        )

        # Task 1 output
        self.room_output = nn.Sequential(
            nn.Linear(shared_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, room_input),
        )

        # Task 2 output
        self.rest_output = nn.Sequential(
            nn.Linear(shared_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, rest_input),
        )

    def forward(self, data):
        hero_input, room_input = data

        hero_x = self.hero_input(hero_input)

        if room_input[0] == 1:
            room_x = self.room_input(room_input[1:])
            rest_x = self.rest_input(torch.zeros(1))
        elif room_input[0] == 2:
            room_x = self.room_input(torch.zeros(2))
            rest_x = self.rest_input(room_input[1:])

        combined_features = torch.cat((hero_x, room_x, rest_x), dim=0)

        x = self.shared_layers(combined_features)

        hero_output = self.hero_output(x)
        room_output = self.room_output(x)
        rest_output = self.rest_output(x)

        return hero_output, room_output, rest_output
