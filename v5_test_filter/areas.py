import dataclasses
from copy import copy
from random import randint

from sklearn.metrics import mean_squared_error

from v5_test_filter.model import GenericModel, StateModel, CombinedNetwork


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
    def __init__(self):
        self.state_model = StateModel(3 + 3, 1)
        self.room_model = CombinedNetwork()
        # health, lvl, gold, room, gold, loose, heal, cost

    def predict(self, hero, rooms: dict):
        all_rooms = []
        for room in rooms.values():
            pred = self.room_model.predict(hero + room)
            diff = Hero(*pred[2][:3])

            short_weight = self.state_model.predict(hero + diff)
            label = None

            all_rooms.append([short_weight.item(), label, pred, room])

        return all_rooms

    def train_once(self, predicted, hero, room, new_hero, new_room):
        original_state = hero + room
        new_state = new_hero + new_room
        diff = [b - a for a, b in zip(original_state, new_state)]
        loss = self.room_model.train_once(original_state, diff)
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
