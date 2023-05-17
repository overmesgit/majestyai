import pygame
import random

import torch

# from room_full import predict_rooms_full, update_room_full_batch, update_room_full
from room_with_types import (
    update_room_full_status, predict_rooms_full_status,
    update_room_full_batch_status,
)
from world import Char, Room

# Game settings
WIDTH = 600
HEIGHT = 600
FPS = 60
TURN_DELAY = 1000

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Turn-based Hero and Monsters")
clock = pygame.time.Clock()

pygame.font.init()
FONT_SIZE = 16
font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)


# Hero class
class Hero(pygame.sprite.Sprite):
    char: Char

    def __init__(self):
        self.char = Char(100, 1, 1, [], [])
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((40, 40))
        self.image.fill("#02781F")
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH / 2, HEIGHT / 2)

    def update(self, x, y, win):
        self.rect.x = x
        self.rect.y = y

        if win:
            self.image.fill("#02781F")
        else:
            self.image.fill("#313978")

        self.rect.clamp_ip(screen.get_rect())

    def draw_text(self, surface, text):
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.center = self.rect.midbottom
        surface.blit(text_surface, text_rect)


# Monster class
class Monster(pygame.sprite.Sprite):
    room: Room

    def __init__(self):
        self.room = Room(random.randint(0, 100), random.randint(0, 5))
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((30, 30))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH // 30) * 30
        self.rect.y = random.randrange(HEIGHT // 30) * 30

    def draw_text(self, surface, text):
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.center = self.rect.center
        surface.blit(text_surface, text_rect)


class Rest(pygame.sprite.Sprite):
    room: Room

    def __init__(self):
        self.room = Room(random.randint(50, 100), 0)
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((30, 30))
        self.image.fill("#2EA60F")
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH // 30) * 30
        self.rect.y = random.randrange(HEIGHT // 30) * 30


# Create sprite groups
all_sprites = pygame.sprite.Group()
monsters = pygame.sprite.Group()
rests = pygame.sprite.Group()
hero = Hero()
all_sprites.add(hero)


def draw_text_bottom(surface, text, font, color=WHITE, margin=10):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.bottom = surface.get_rect().bottom - margin
    text_rect.x = margin
    surface.blit(text_surface, text_rect)


def player_help(rooms: list[Room], rests: list[Room], character: Char, value):
    room_data = [(1, 0, room.level, room.gold, character.level, character.power) for room in rooms]
    rest_data = [(0, 1, r.level, r.gold, character.level, character.power) for r in rests]
    room_data += rest_data
    room_inputs, predictions = predict_rooms_full_status(room_data)
    character.room_memory.append((room_inputs[0], value))
    character.room_memory = character.room_memory[-20:]
    print(character.room_memory)
    # update_room_full_batch(character.room_memory)


def play_game(rooms: list[Room], rests: list[Room], character: Char):
    room_data = [(1, 0, room.level, room.gold, character.level, character.power) for room in rooms]
    rest_data = [(0, 1, r.level, r.gold, character.level, character.power) for r in rests]

    room_data += rest_data
    room_inputs, predictions = predict_rooms_full_status(room_data)
    for r, p in zip(room_data, predictions):
        print(f'{r}: {p}')

    print(predictions.sum(dim=1))
    chosen_room = torch.argmax(predictions.sum(dim=1))
    print(f"chosen room: {chosen_room + 1}")

    # Determine if the character won
    # TODO: clean problems with rooms and rests
    if chosen_room < len(rooms):
        current_room = rooms[chosen_room]
        if character.level <= 0:
            actual_win_prob = 0
        else:
            actual_win_prob = 0
            if character.power * character.level > current_room.level:
                actual_win_prob = 1

        win = actual_win_prob > 0
        print(f"{actual_win_prob}", "WIN!" if win else "DEFEAT!")

        got_money = current_room.gold
        if win:
            prior_prob = got_money / 5
        else:
            prior_prob = 0

        character.level = max(character.level - round(current_room.level / 2), 0)
        if win:
            character.gold += current_room.gold
            character.power += 0.05
    else:
        win = False
        current_room = rests[chosen_room - len(rooms)]
        max_heal = 100 - character.level
        healed = min(max_heal, current_room.level)
        # max: 100 / 100 => 1
        prior_prob = healed / 100

        print(f"Healed {character.level} => {character.level + healed}")
        character.level += healed

    print(f"Priority result {prior_prob}")

    # character.room_memory.append((room_inputs[chosen_room], prior_prob))
    # update_room_full(room_inputs[chosen_room], prior_prob)
    # update_room_full_batch(character.room_memory)

    character.room_memory.append((room_inputs[chosen_room], [character.level, character.power, character.gold]))
    # update_room_full_status(room_inputs[chosen_room],
    #                         [character.level, character.power, character.gold])
    update_room_full_batch_status(character.room_memory)

    return chosen_room, win


# Game loop
running = True
last_turn_time = pygame.time.get_ticks()
turn = 1


def add_monster(new_monster):
    all_sprites.add(new_monster)
    monsters.add(new_monster)


def add_rest(new_rest):
    all_sprites.add(new_rest)
    rests.add(new_rest)


for i in range(3):
    add_monster(Monster())

while running:
    clock.tick(FPS)
    turn_requested = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                turn_requested = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click_position = event.pos
            clicked_monster, clicked_rest = None, None
            for monster in monsters:
                if monster.rect.collidepoint(click_position):
                    clicked_monster = monster
                    break
            for rest in rests:
                if rest.rect.collidepoint(click_position):
                    clicked_rest = rest
                    break

            if clicked_monster:
                if event.button == 1:  # Left click
                    clicked_monster.image.fill("#3CFF4F")
                    player_help([clicked_monster.room], [], hero.char, 1)
                elif event.button == 3:  # Right click
                    clicked_monster.image.fill("#FFCB5F")
                    player_help([clicked_monster.room], [], hero.char, 0)
            if clicked_rest:
                if event.button == 1:  # Left click
                    clicked_rest.image.fill("#4FA622")
                    player_help([], [clicked_rest.room], hero.char, 1)

    current_time = pygame.time.get_ticks()
    if turn_requested:
        last_turn_time = current_time
        turn_requested = False
        turn += 1

        rooms = [m.room for m in monsters]
        rooms_for_rest = [r.room for r in rests]
        if rooms:
            open_room, win = play_game(rooms, rooms_for_rest, hero.char)

            if open_room < len(rooms):
                m = monsters.sprites()[open_room]
                hero.update(m.rect.x, m.rect.y, win)

                if win:
                    monsters.remove(m)
                    all_sprites.remove(m)
            else:
                m = rests.sprites()[open_room - len(rooms)]
                hero.update(m.rect.x, m.rect.y, win)
                rests.remove(m)
                all_sprites.remove(m)

        # Spawn monsters
        if len(monsters) < 5:
            add_monster(Monster())

        if len(rooms_for_rest) < 1:
            add_rest(Rest())

        # Check for collisions
        # hit_monsters = pygame.sprite.spritecollide(hero, monsters, True)
        # for monster in hit_monsters:
        #     print("Monster defeated!")

    # Draw
    screen.fill((0, 0, 0))
    all_sprites.draw(screen)

    # Draw text over monsters
    for monster in monsters:
        monster.draw_text(screen, f"{monster.room.level}-{monster.room.gold}")
    hero.draw_text(screen, f"{hero.char.level}-{round(hero.char.power, 2)}")
    draw_text_bottom(screen, f"Turn {turn}", font, WHITE)
    pygame.display.flip()

pygame.quit()
