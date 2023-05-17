import pygame


class EnvMap:
    scale = 5
    screen_width = 800
    screen_height = 800

    def __init__(self):
        self.screen = None
        self.clock = None

    def render(self, state):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            self.screen.fill((0, 100, 0))

        RED = (255, 0, 0)
        for obj in state:
            pygame.draw.rect(self.screen, RED, (obj[0]*self.scale, obj[1]*self.scale, self.scale, self.scale))

        # Update the display
        pygame.display.flip()


if __name__ == '__main__':
    EnvMap().render([[80, 80]])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.update()

    # Quit Pygame
    pygame.quit()
