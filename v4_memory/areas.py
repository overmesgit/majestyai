import dataclasses
from copy import copy
from random import randint

from sklearn.cluster import MiniBatchKMeans

from v3_multi_nets.model import GenericModel, StateModel


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
    hp: int = 1
    gold: int = 1

    @classmethod
    def random_room(cls):
        return Room(randint(10, 99), randint(0, 5))

    def to_norm_array(self):
        return [self.hp / 100, self.gold / 5]

    def to_array(self):
        return [self.hp, self.gold]

    def process(self, hero):
        old_room = copy(self)
        old_hero = copy(hero)
        if self.hp < hero.hp:
            hero.gold += self.gold
            hero.level += 0.1

            self.hp = 0
            self.gold = 0
        else:
            self.hp = max(self.hp - hero.hp / 2, 0)
            hero.gold = max(hero.gold - 3, 0)

        hero.hp = max(hero.hp - old_room.hp / 2, 0)
        return old_room, old_hero


@dataclasses.dataclass
class Rest(Base):
    hp: int = 1
    cost = 1
    min_value = 50
    max_value = 100

    @classmethod
    def random_rest(cls):
        return cls(randint(cls.min_value, cls.max_value))

    def to_array(self):
        return [self.hp]

    def to_norm_array(self):
        return [self.hp / 100]

    def process(self, hero):
        old_room = copy(self)
        old_hero = copy(hero)
        if hero.gold >= self.cost:
            hero.hp = min(hero.hp + self.hp, 100)
            hero.gold -= self.cost
            self.hp = randint(self.min_value, self.max_value)
        return old_room, old_hero


@dataclasses.dataclass
class Home(Rest):
    hp: int = 25
    cost = 0
    min_value = 10
    max_value = 50


@dataclasses.dataclass
class Mine(Base):
    hp: int = 10

    def to_array(self):
        return []

    def to_norm_array(self):
        return []

    def process(self, hero):
        old_room = copy(self)
        old_hero = copy(hero)
        if hero.hp >= 10:
            hero.hp -= 10
            hero.gold += 1
        return old_room, old_hero


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


class WorldModel:
    def __init__(self):
        self.state_model = StateModel(3, 1)

        self.long_memory = []

        self.clusters = 10
        self.kmeans = MiniBatchKMeans(n_clusters=self.clusters, batch_size=10)
        for i in range(self.clusters):
            self.long_memory.append(StateModel(3, 1, rate=0.001))

        self.room_to_model = {
            Room: GenericModel(3 + 2, 3),
            Rest: GenericModel(3 + 1, 3),
            Home: GenericModel(3 + 1, 3),
            Mine: GenericModel(3, 3),
        }

        self.fitted = False
        # self.state_model.load_state_dict(torch.load('state_model.pth'))
        # self.state_model.eval()
        self.memory = []

    def predict(self, hero, rooms: dict):
        all_rooms = []
        for room in rooms.values():
            pred = self.room_to_model[type(room)].predict(hero + room)
            diff = Hero(*pred[:3].tolist())
            short_weight = self.state_model.predict(diff.to_array())
            long_mem, label = (0,), None
            if self.fitted:
                label = self.kmeans.predict([diff.to_norm_array()])[0]
                long_mem = self.long_memory[label].predict(diff.to_array())

            weight = (0.5 * short_weight[0], 0.5 * long_mem[0])
            all_rooms.append([weight, label, pred, room])
        return all_rooms

    def train_once(self, predicted, hero, room, new_hero, new_room):
        diff_h = hero.diff(new_hero)
        loss = self.room_to_model[type(room)].train_once(hero + room, diff_h.to_array())
        state_loss = self.train_state(predicted, hero, room, new_hero, new_room)
        return loss, state_loss

    def train_state(self, predicted, hero, room, new_hero, new_room):
        # if cool experience update stat model
        health = normalize(new_hero.hp - hero.hp, -100, 100)
        gold = normalize(new_hero.gold - hero.gold, -5, 5)

        if new_hero.hp == 0:
            weight = 0
        elif new_hero.hp == hero.hp and new_hero.gold <= hero.gold:
            weight = 0
        else:
            weight = 0.5 * health + 0.5 * gold

        diff = hero.diff(new_hero)
        print(f"Weight: {weight} {health} {gold}")

        self.memory.append((weight, hero, diff))
        if len(self.memory) > self.clusters:
            self.kmeans.partial_fit([d.to_norm_array() for _, h, d in self.memory])
            self.fitted = True
            self.memory = []

        if self.fitted:
            label = self.kmeans.predict([diff.to_norm_array()])[0]
            print(f"Label {label}")
            self.long_memory[0].train_once(diff.to_array(), [weight])

        # boredom to prevent stack in the same place
        # may be only gather uniq states in long memory
        # 0.5 memory + 0.5 prediction
        return self.state_model.train_once(diff.to_array(), [weight])

    def some_memory(self):
        pass

        # self.memory.append((weight, (hero, diff)))
        # if len(self.memory) > 5:
        #     max_mem = max(self.memory[:10], key=lambda w: w[0])
        #     min_mem = min(self.memory[:10], key=lambda w: w[0])
        #     print(max_mem, min_mem)
        #     # self.state_model = StateModel(6, 1)
        #     for (w, (h, d)) in [max_mem, min_mem]:
        #         # if (d.hp, d.gold) not in self.diff_memory or abs(weight - predicted) > 0.1:
        #         # self.diff_memory[(d.hp, d.gold)] = True
        #         self.long_memory.train_once(h + d, [w])
        #         # self.long_memory.train_once(h + d, [w])
        #         # self.state_model.train_once(h + d, [w])
        #     self.memory = []

        # for (w, (h, d)) in self.top_actions + self.worst_actions:
        #     self.state_model.train_once(h + d, [w])

        # if len(self.top_actions) < 4 or any([weight > w for (w, _) in self.top_actions]):
        #     self.top_actions.append((weight, (hero, diff)))
        #     if len(self.top_actions) > 4:
        #         self.top_actions = self.top_actions[1:]
        #
        # if len(self.worst_actions) < 4 or any([weight < w for (w, _) in self.worst_actions]):
        #     self.worst_actions.append((weight, (hero, diff)))
        #     if len(self.worst_actions) > 4:
        #         self.worst_actions = self.worst_actions[1:]


@dataclasses.dataclass
class Hero(Base):
    hp: int
    level: float
    gold: int
    normalized: bool = False

    def to_norm_array(self):
        if not self.normalized:
            return [self.hp / 100, self.level / 100, self.gold / 5]
        else:
            return self.to_array()

    def to_array(self):
        return [self.hp, self.level, self.gold]
