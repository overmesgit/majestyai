import random

import arcade

class HeroS(arcade.Sprite):
    def __init__(self, *args, data, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

    def draw(self, *args, **kwargs):
        super().draw(*args, **kwargs)
        text = f"{self.data.hp}/{self.data.gold}/{self.data.level:.1f}"
        x = self.center_x
        y = self.center_y
        color = arcade.color.RED
        font_size = 16
        width = 100
        align = "center"
        anchor_x = "center"
        anchor_y = "center"
        arcade.draw_text(text, x, y, color, font_size, width, align,
                         anchor_x=anchor_x,
                         anchor_y=anchor_y)


class MonsterS(arcade.Sprite):
    def __init__(self, *args, data, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    def draw(self, *args, **kwargs):
        super().draw(*args, **kwargs)
        text = f"{self.data.hp}/{self.data.gold}"
        x = self.center_x
        y = self.center_y
        color = arcade.color.RED
        font_size = 16
        width = 100
        align = "center"
        anchor_x = "center"
        anchor_y = "center"
        arcade.draw_text(text, x, y, color, font_size, width, align,
                         anchor_x=anchor_x,
                         anchor_y=anchor_y)


class RestS(arcade.Sprite):
    def __init__(self, *args, data, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    def draw(self, *args, **kwargs):
        super().draw(*args, **kwargs)
        text = f"{self.data.hp}"
        x = self.center_x
        y = self.center_y
        color = arcade.color.RED
        font_size = 16
        width = 100
        align = "center"
        anchor_x = "center"
        anchor_y = "center"
        arcade.draw_text(text, x, y, color, font_size, width, align,
                         anchor_x=anchor_x,
                         anchor_y=anchor_y)
