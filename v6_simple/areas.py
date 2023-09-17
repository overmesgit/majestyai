import dataclasses
import time
from copy import copy
from random import randint

from sklearn.metrics import mean_squared_error

from v6_simple.model import GenericModel, StateModel


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
    loose: float = 0.03

    @classmethod
    def random_room(cls):
        return Room(randint(10, 99) / 100, randint(0, 5) / 100)

    def to_array(self):
        return [self.hp, self.gold, self.loose, 0, 0]

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
            hero.gold = max(hero.gold - self.loose, 0)

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
        return [0, 0, 0, self.hp, self.cost]

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
    hp: float = 0.1
    gold: float = 0.01

    def to_array(self):
        return [self.hp, self.gold, 0, 0, 0]

    def to_norm_array(self):
        return []

    def process(self, hero):
        old_room = copy(self)
        old_hero = copy(hero)
        if hero.hp >= self.hp:
            hero.hp -= self.hp
            hero.gold += self.gold
        return old_room, old_hero


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


class WorldModel:
    @dataclasses.dataclass
    class Memo:
        hero: Base
        room: Base
        new_hero: Base
        new_room: Base
        weight: float

    @dataclasses.dataclass
    class Prediction:
        weight: float
        room_pred: list
        room: Base
        params: dict

    def __init__(self):
        self.state_model = StateModel(3 + 3, 1)
        self.overall_model = GenericModel(8, 16, 0.0001)

        # health, lvl, gold, room, gold, loose, heal, cost
        self.state_memory: list[WorldModel.Memo] = []
        self.memo_recount = 0
        self.data_file = open(f'stat/d{time.time()}.txt', 'w')


    def add_memory(self, memo: 'Memo'):
        self.memo_recount += 1

        if self.memo_recount < 10:
            self.state_memory.append(memo)
        else:
            self.state_memory.append(memo)

    def search_memory(self, hero, room) -> 'Memo':
        min_state = (100, None)
        mean_avg = 0
        mean_count = 0
        for m in self.state_memory[:30]:
            mean = mean_squared_error(hero + room, m.hero + m.room)
            mean_avg += mean
            mean_count += 1
            if mean < min_state[0]:
                min_state = (mean, m)
        return min_state[1], mean_avg / (mean_count or 1)

    def predict(self, hero, rooms: dict) -> list[Prediction]:
        all_rooms = []
        for room in rooms.values():
            pred = self.overall_model.predict(hero + room)
            diff = Hero(*pred[:3].tolist())

            memory_hero_room, mean_avg = self.search_memory(hero, room)
            long_mem = 0.5
            if memory_hero_room:
                memo_diff = hero.diff(memory_hero_room.new_hero)
                diff = Hero(*[a + b / 2 for a, b in zip(pred[:3].tolist(), memo_diff.to_array())])
                long_mem = memory_hero_room.weight

            state_pred = self.state_model.predict(hero + diff)

            state_weight = sum(0.5 * state_pred, 0.5 * long_mem)

            all_rooms.append(
                WorldModel.Prediction(state_weight, pred, room,
                                      {'mem': long_mem, 'net': state_pred.item()}))

        return all_rooms

    def train_once(self, predicted, hero, room, new_hero, new_room):
        original_state = hero + room
        new_state = new_hero + new_room
        diff = [new_st - orig_st for orig_st, new_st in zip(original_state, new_state)]
        self.data_file.write(f"{','.join(original_state)}, {','.join(diff)}\n")

        memory_hero_room, mean_avg = self.search_memory(hero, room)
        print('avg', mean_avg)
        loss = self.overall_model.train_once(original_state, diff)
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

        self.add_memory(WorldModel.Memo(hero, room, new_hero, new_room, weight))

        diff = hero.diff(new_hero)
        print(f"Weight: {weight} {health} {gold}")

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
