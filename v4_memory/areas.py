import dataclasses
from copy import copy
from random import randint

from sklearn.metrics import mean_squared_error

from v4_memory.model import GenericModel, StateModel


class Base:

    def to_array(self) -> list:
        pass

    def to_norm_array(self) -> list:
        pass

    def diff(self, other: 'Base') -> 'Base':
        return self.__class__(*[b - a for a, b in zip(self.to_array(), other.to_array())])

    def __add__(self, other: 'Base') -> list:
        return self.to_array() + other.to_array()

    def process(self, hero: 'Base') -> tuple['Base', 'Base']:
        pass


@dataclasses.dataclass
class Room(Base):
    hp: float = 1
    gold: float = 1

    @classmethod
    def random_room(cls):
        return Room(randint(10, 99) / 100, randint(0, 5) / 100)

    def to_array(self):
        return [self.hp, self.gold]

    def process(self, hero: 'Hero'):
        old_room = copy(self)
        old_hero = copy(hero)
        if self.hp < hero.hp:
            hero.gold += self.gold
            hero.level += 0.1

            self.hp = 0
            self.gold = 0
        else:
            self.hp = max(self.hp - hero.hp / 2, 0)
            hero.gold = max(hero.gold - 0.03, 0)

        hero.hp = round(max(hero.hp - old_room.hp / 2, 0), 2)
        hero.gold = round(hero.gold, 2)
        return old_room, old_hero


@dataclasses.dataclass
class Rest(Base):
    hp: float = 1
    cost = 0.01
    min_value = 50
    max_value = 100

    @classmethod
    def random_rest(cls):
        return cls(randint(cls.min_value, cls.max_value) / 100)

    def to_array(self):
        return [self.hp]

    def process(self, hero):
        old_room = copy(self)
        old_hero = copy(hero)
        if hero.gold >= self.cost:
            hero.hp = min(hero.hp + self.hp, 1)
            hero.gold -= self.cost
            self.hp = randint(self.min_value, self.max_value) / 100
        return old_room, old_hero


@dataclasses.dataclass
class Home(Rest):
    hp: float = 0.25
    cost = 0
    min_value = 10
    max_value = 50


@dataclasses.dataclass
class Mine(Base):
    hp: int = 0.1

    def to_array(self):
        return []

    def to_norm_array(self):
        return []

    def process(self, hero):
        old_room = copy(self)
        old_hero = copy(hero)
        if hero.hp >= self.hp:
            hero.hp -= self.hp
            hero.gold += 0.01
        return old_room, old_hero


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


class WorldModel:
    def __init__(self):
        self.state_model = StateModel(3 + 3, 1)

        self.state_memory = {
            Room: [],
            Rest: [],
            Home: [],
            Mine: [],
        }

        self.room_to_model = {
            Room: GenericModel(3 + 2, 3 + 2),
            Rest: GenericModel(3 + 1, 3 + 1),
            Home: GenericModel(3 + 1, 3 + 1),
            Mine: GenericModel(3, 3),
        }

    def search_memory(self, hero, room) -> list:
        memory_rooms = self.state_memory[type(room)]
        min_state = (100, None)
        for m in memory_rooms:
            mean = mean_squared_error(hero + room, m[0])
            if mean < min_state[0]:
                min_state = (mean, m)
        return min_state[1]

    def predict(self, hero, rooms: dict):
        all_rooms = []
        for room in rooms.values():
            pred = self.room_to_model[type(room)].predict(hero + room)
            new_hero = Hero(*pred[:3].tolist())
            # how to make memory smaller
            memory_hero_room = self.search_memory(hero, room)
            long_mem = 0
            if memory_hero_room:
                avg = [a + b / 2 for a, b in zip(pred[:3].tolist(), memory_hero_room[1][:3])]
                new_hero = Hero(*avg)
                long_mem = memory_hero_room[2]

            diff = hero.diff(new_hero)
            short_weight = self.state_model.predict(hero + diff)
            label = None

            weight = (0.5 * short_weight[0], 0.5 * long_mem)
            all_rooms.append([weight, label, pred, room])

        return all_rooms

    def train_once(self, predicted, hero, room, new_hero, new_room):
        loss = self.room_to_model[type(room)].train_once(hero + room, new_hero + new_room)
        state_loss = self.train_state(predicted, hero, room, new_hero, new_room)
        return loss, state_loss

    def train_state(self, predicted, hero, room, new_hero, new_room):
        # if cool experience update stat model
        health = normalize(new_hero.hp - hero.hp, -1, 1)
        gold = normalize(new_hero.gold - hero.gold, -1, 1)

        if new_hero.hp == 0:
            weight = 0
        elif new_hero.hp == hero.hp and new_hero.gold <= hero.gold:
            weight = 0
        else:
            weight = 0.5 * health + 0.5 * gold

        diff = hero.diff(new_hero)
        print(f"Weight: {weight} {health} {gold}")

        self.state_memory[type(room)].append((hero + room, new_hero + new_room, weight))

        # boredom to prevent stack in the same place
        # may be only gather uniq states in long memory
        # 0.5 memory + 0.5 prediction
        return self.state_model.train_once(hero + diff, [weight])


@dataclasses.dataclass
class Hero(Base):
    hp: float
    level: float
    gold: float

    @classmethod
    def new_hero(cls):
        return Hero(1, 1, 0)

    def to_array(self):
        return [self.hp, self.level, self.gold]
