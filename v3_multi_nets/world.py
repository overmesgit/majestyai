import time
from collections import Counter

from v3_multi_nets.areas import Hero, Room, Rest, Home, Mine, WorldModel


class World:
    model = WorldModel()

    def __init__(self):
        self.turn = 0
        self.rooms = dict()
        for i in range(3):
            self.add_monster()

        room = Rest.random_rest()
        self.rooms[id(room)] = room
        room = Home.random_rest()
        self.rooms[id(room)] = room
        room = Mine()
        self.rooms[id(room)] = room
        self.hero = Hero(100, 1, 1)
        self.hero_history = []
        self.data_file = open(f'stat/{time.time()}.txt', 'w')
        self.data_file.write('turn,gold\n')

    def add_monster(self):
        room = Room.random_room()
        self.rooms[id(room)] = room

    def gold_per_turn(self):
        if self.hero_history:
            gold = [x.gold for x in self.hero_history]
            return sum(gold[-10:]) / 10, sum(gold) / len(self.hero_history)
        else:
            return 0, 0

    def do_turn(self, next_action_id=None):
        self.turn += 1
        all_rooms = self.model.predict(self.hero, self.rooms)

        for r in all_rooms:
            print(f'{r[0][0]:.2f} {r[0][1]:.2f} {r[1]} {r[2].tolist()} {r[3]}')

        predicted = None
        if next_action_id and next_action_id in self.rooms:
            room = self.rooms[next_action_id]
            print('Use players room.')
        else:
            max_room = max(all_rooms, key=lambda a: sum(a[0]))
            room = max_room[3]
            predicted = sum(max_room[0])

        print(self.hero)
        print(room)

        old_room, old_hero = room.process(self.hero)
        self.hero_history.append(self.hero)
        self.data_file.write(f"{self.turn},{self.hero.gold}\n")

        loss = self.model.train_once(predicted, old_hero, old_room, self.hero, room)

        if room.hp <= 0:
            del self.rooms[id(room)]

        counter = Counter([type(r) for r in self.rooms.values()])
        if counter[Room] < 5:
            self.add_monster()

        print(f'Hero: {self.hero} Loss: {loss}')
        return room, self.hero, self.rooms


if __name__ == '__main__':
    w = World()

    # while input('q for exit:') != 'q':
    #     w.do_turn()

    while True:
        w.do_turn()
