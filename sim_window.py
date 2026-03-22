import pygame
import math
from network import Agent, Network
from util import invert_color
import numpy as np


class SimWindow:
    def __init__(self, size: tuple):
        self.size = size

        pygame.init()
        self.background_color = (234, 212, 252)

        self.font = pygame.font.SysFont("Consolas", 12)  # monospace
        self.info_text_surface = None

        self.screen = pygame.display.set_mode([self.size[0], self.size[1]])
        pygame.display.set_caption("sim")
        self.screen.fill(self.background_color)
        pygame.display.flip()

        self.origin = (0, 0)
        self.shamt = (0, 0)

    def clear(self):
        self.screen.fill(self.background_color)

    def draw(self, network: Network, info_text: bool = False, color_dummy=None):
        agents = network.agents
        edges = network.edges

        if color_dummy is None:
            self.shamt = self.shift_amount(agents)
            self.draw_grid(self.shamt)

        for agent in agents:
            footprint = agent.get_footprint()
            screen_coords = [
                (x[0] - self.shamt[0], self.size[1] - (x[1] - self.shamt[1]))
                for x in footprint
            ]
            pygame.draw.polygon(
                self.screen,
                color_dummy if color_dummy is not None else invert_color(self.background_color),
                screen_coords,
            )

        for i, j in edges:
            pos_i = agents[i].pose.position[:2] - self.shamt
            pos_j = agents[j].pose.position[:2] - self.shamt
            screen_i = (pos_i[0], self.size[1] - pos_i[1])
            screen_j = (pos_j[0], self.size[1] - pos_j[1])
            pygame.draw.line(
                self.screen,
                color_dummy if color_dummy is not None else invert_color(self.background_color),
                screen_i,
                screen_j,
            )

        if info_text or self.info_text_surface:
            self.screen.blit(self.info_text_surface, (5, 5))


    def flip(self):
        pygame.display.flip()

    def draw_grid(self, shamt, spacing=25):
        color = (200, 200, 200)

        width, height = self.size

        x_start = int(shamt[0] // spacing * spacing)
        y_start = int(shamt[1] // spacing * spacing)

        x = x_start
        while x < shamt[0] + width:
            screen_x = int(x - shamt[0])
            pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, height))
            x += spacing

        y = y_start
        while y < shamt[1] + height:
            screen_y = int(y - shamt[1])
            pygame.draw.line(self.screen, color, (0, screen_y), (width, screen_y))
            y += spacing

    def shift_amount(self, obj_list: list[Agent]):
        com = np.zeros(2)
        for obj in obj_list:
            com += obj.pose.position[:2]
        com /= len(obj_list)

        shift_amount = (com[0] - self.size[0] / 2, com[1] - self.size[1] / 2)
        return shift_amount

    def get_events(self):
        return pygame.event.get()

    def set_info_text(self, s: str, color: tuple):
        self.info_text_surface = self.font.render(s, True, color)

    def quit(self):
        pygame.display.quit()
        pygame.quit()

    def handle_events(self, events):
        terminate = False
        ret = None
        for e in events:
            if e.type == pygame.QUIT:
                terminate = True
            if e.type == pygame.KEYDOWN:

                # quit
                if e.key == pygame.K_ESCAPE or e.key == pygame.K_q:
                    terminate = True

                # snapshot plot
                if e.key == pygame.K_SPACE:
                    ret = "plot"
                    terminate = True
        return terminate, ret
