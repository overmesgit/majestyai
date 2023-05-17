import random
import torch

from room import predict_rooms, update_room_batch
from prior import predict_prior, update_prior_batch
from world import timeit, Char, World


@timeit
def play_game(character: Char):
    w = World()

    room_inputs, predictions = predict_rooms(w.rooms, character)
    prior_inputs, prior_predictions = predict_prior(w.rooms, predictions)

    w.print_stage(character, predictions, prior_predictions)

    # Combine player advice and predictions
    final_predictions = prior_predictions.clone()

    # Character chooses a room
    chosen_room = torch.argmax(final_predictions)
    print(f"chosen room: {chosen_room + 1}")

    # Determine if the character won
    current_room = w.rooms[chosen_room]
    actual_win_prob = character.level / (current_room.level + character.level)

    random_random = random.random()
    win = random_random < actual_win_prob
    print(f"{random_random} < {actual_win_prob}", "WIN!" if win else "DEFEAT!")

    # train priority
    got_money = current_room.gold
    if win:
        prior_prob = 0.5 * (predictions[chosen_room] + got_money / 5)
    else:
        prior_prob = 0.5 * predictions[chosen_room]
    print(f"Priority result {prior_prob}")

    character.room_memory.append((room_inputs[chosen_room], actual_win_prob))
    character.prior_memory.append((prior_inputs[chosen_room], prior_prob))

    update_room_batch(character.room_memory)
    update_prior_batch(character.prior_memory)

    if win:
        character.level -= current_room.level / 2
    else:
        character.level = random.randint(1, 100)

    character.gold += current_room.gold

    return character


character_level = Char(100, 0, [], [])
count = 0
while count < 15:
    character_level = play_game(character_level)
    count += 1
