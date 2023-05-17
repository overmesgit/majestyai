from copy import copy

import arcade
import random

from v3_multi_nets.areas import Room, Rest, Home, Mine
from v3_multi_nets.ui_class import HeroS, MonsterS, RestS
from v3_multi_nets.world import World

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Hero vs Monsters"

HERO_SPEED = 5
MONSTER_SPEED = 2
NUM_MONSTERS = 5


class Game(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.next_action = None
        self.hero = None
        self.monsters = None
        self.rests = None
        self.active = False
        self.run = False

        self.world = World()

    def setup(self):
        arcade.set_background_color(arcade.color.BLUE_GRAY)

        self.hero = HeroS("img/hero.png", scale=1, image_width=96, image_height=96,
                          data=self.world.hero)
        self.hero.center_x = SCREEN_WIDTH // 2
        self.hero.center_y = SCREEN_HEIGHT // 2

        self.rests = arcade.SpriteList()
        self.monsters = arcade.SpriteList()

        self.sync_sprites(self.world.rooms, {})

    def add_monster(self, data):
        monster = MonsterS("img/monster.png", scale=1, image_width=96, image_height=96, data=data)
        monster.center_x = random.randint(50, 200)
        monster.center_y = random.randint(50, 600)
        self.monsters.append(monster)

    def add_rest(self, data):
        rest = RestS("img/house.png", scale=0.3, image_width=512, image_height=512, data=data)
        rest.center_x = SCREEN_WIDTH // 2 + 200
        rest.center_y = SCREEN_HEIGHT // 2
        self.rests.append(rest)

    def add_home(self, data):
        rest = RestS("img/house.png", scale=0.2, image_width=512, image_height=512, data=data)
        rest.center_x = SCREEN_WIDTH // 2 + 200
        rest.center_y = SCREEN_HEIGHT // 2 + 200
        self.rests.append(rest)

    def add_mine(self, data):
        rest = RestS("img/mine.png", scale=0.2, image_width=512, image_height=512, data=data)
        rest.center_x = SCREEN_WIDTH // 2 + 300
        rest.center_y = SCREEN_HEIGHT // 2 + 200
        self.rests.append(rest)

    def draw_stats(self):
        ten, all = self.world.gold_per_turn()
        text = f"{self.world.turn} {all:.2f} {ten:.2f}"
        x = SCREEN_WIDTH - 100
        y = 10
        color = arcade.color.GOLD
        font_size = 16
        width = 200
        align = "center"
        arcade.draw_text(text, x, y, color, font_size, width, align,
                         anchor_x=align,
                         anchor_y=align)

    def on_draw(self):
        arcade.start_render()
        for m in self.monsters:
            m.draw()

        for r in self.rests:
            r.draw()

        self.draw_stats()
        self.hero.draw()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.active = not self.active

        if key == arcade.key.LCTRL:
            self.run = not self.run

    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            # Check if any sprite was clicked
            for sprite in list(self.monsters) + list(self.rests):
                if sprite.collides_with_point((x, y)):
                    print(f"Clicked on sprite: {sprite}")
                    self.next_action = sprite
                    break

    def sync_sprites(self, rooms, sprites):
        for i, r in sprites.items():
            if i not in rooms:
                if isinstance(r, MonsterS):
                    self.monsters.remove(r)
                elif isinstance(r, RestS):
                    self.rests.remove(r)

        for i, v in rooms.items():
            if i not in sprites:
                if isinstance(v, Room):
                    self.add_monster(v)
                elif isinstance(v, Mine):
                    self.add_mine(v)
                elif isinstance(v, Home):
                    self.add_home(v)
                elif isinstance(v, Rest):
                    self.add_rest(v)

    def update(self, delta_time):
        # train state model
        if not self.active:
            return

        if not self.run:
            self.active = False

        next_action = id(self.next_action.data) if self.next_action else None
        room, hero, rooms = self.world.do_turn(next_action)
        self.next_action = None

        self.hero.data = hero

        sprites = {}
        for r in list(self.monsters) + list(self.rests):
            sprites[id(r.data)] = r

        self.sync_sprites(rooms, sprites)

        selected_room = sprites[id(room)]
        self.hero.center_x = selected_room.center_x + 10
        self.hero.center_y = selected_room.center_y - 10

        self.hero.update()
        self.monsters.update()


if __name__ == "__main__":
    game = Game()
    game.setup()
    arcade.run()
