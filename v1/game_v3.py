import random
import torch

from room_full import predict_rooms_full, update_room_full_batch
from world import timeit, Char, World


@timeit
def play_game(character: Char):
    w = World()
    room_data = [(room.level, room.gold, character.level) for room in w.rooms]
    room_inputs, predictions = predict_rooms_full(
        room_data, character)

    w.print_stage_predict(character, predictions)

    # Character chooses a room
    chosen_room = torch.argmax(predictions)
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
        prior_prob = got_money / 5
    else:
        prior_prob = 0
    print(f"Priority result {prior_prob}")

    character.room_memory.append((room_inputs[chosen_room], actual_win_prob))

    update_room_full_batch(character.room_memory)

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
