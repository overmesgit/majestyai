import dataclasses
import functools
import random
import time

import torch


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to execute.")
        return result

    return wrapper


@dataclasses.dataclass
class Room:
    level: int
    gold: int


@dataclasses.dataclass
class Char:
    level: int
    power: int
    gold: int
    room_memory: list[tuple[torch.Tensor, float]]
    prior_memory: list[tuple[torch.Tensor, float]]


class World:
    rooms: list[Room]

    def __init__(self):
        # Generate 3 random rooms
        self.rooms = [Room(random.randint(1, 100), random.randint(1, 5)) for _ in range(3)]
        # safe room
        # safe_room = 3
        # rooms += [Room(0, 0)]

    def print_stage_predict(self, character: Char, predictions):
        print(f"my level {character.level} my money {character.gold}")
        for i, (r, p) in enumerate(zip(self.rooms, predictions)):
            print(f"room {i + 1}: level {r} prediction prob: {p}")

    def print_stage(self, character: Char, predictions, prior_predictions):
        print(f"my level {character.level} my money {character.gold}")
        for i, (r, p, prior) in enumerate(zip(self.rooms, predictions, prior_predictions)):
            print(f"room {i + 1}: level {r} prediction win: {p} choose: {prior}")
