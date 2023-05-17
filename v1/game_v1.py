import random
import torch

from room import predict_rooms, update_rooms
from prior import predict_prior, update_prior
from world import timeit, Char, World


@timeit
def play_game(character: Char):
    w = World()

    room_inputs, predictions = predict_rooms(w.rooms, character)
    # prior_inputs, prior_predictions = predict_prior(w.rooms, predictions)

    w.print_stage(character, predictions, predictions)

    # Player advises a room (index)
    # player_advice = get_user_input()

    # Combine player advice and predictions
    final_predictions = predictions.clone()
    # if player_advice < 3:
    #     final_predictions[player_advice] += 0.2

    # action_inputs, action = predict_action(max(final_predictions))
    # if action == 1:
    #     print("Rest")
    #     character.level = min(100, character.level + 50)
    #     update_action(action_inputs, 1 if character.level < 100 else 0)
    #     # update_action(action_inputs, 1 if player_advice == 4 else 0)
    #     return character

    # Character chooses a room
    chosen_room = torch.argmax(final_predictions)
    print(f"chosen room: {chosen_room + 1}")

    # Determine if the character won
    current_room = w.rooms[chosen_room]
    actual_win_prob = character.level / (current_room.level + character.level)

    random_random = random.random()
    win = random_random < actual_win_prob
    print(f"{random_random} < {actual_win_prob}", "WIN!" if win else "DEFEAT!")

    update_rooms(actual_win_prob, room_inputs[chosen_room])

    if win:
        character.level -= current_room.level / 2
    else:
        character.level = random.randint(1, 100)

    character.gold += current_room.gold
    return character


character_level = Char(100, 0)
count = 0
while count < 100:
    character_level = play_game(character_level)
    count += 1
