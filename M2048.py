import gymnasium as gym
from gymnasium import spaces
import numpy as np
from random import randrange
import random
import pygame
import pygame.freetype
import time
import torch


class M2048(gym.Env):
    def __init__(self, sqr=4, seed=None, silent_mode=True):
        self.sqr = sqr
        self.map = np.zeros((sqr, sqr), dtype=int)

        self.block_size = 100
        self.window_height = self.block_size * sqr
        self.window_width = self.block_size * sqr

        self.generate_tile()
        self.generate_tile()

        self.screen = None
        if not seed:
            seed = time.time()
        random.seed(seed)
        self.silent_mode = silent_mode

        self.observation_space = spaces.Box(
            0, 2 << 15, shape=self.get_obs().shape, dtype=int
        )
        self.action_space = spaces.Discrete(4)

    def get_obs(self):
        return self.map.flatten()

    def generate_tile(self):
        empty_cells = np.where(self.map == 0)
        chioce = random.choice(range(len(empty_cells[0])))
        assert len(empty_cells) > 0
        row, col = empty_cells[0][chioce], empty_cells[1][chioce]
        self.map[row][col] = 4 if random.random() < 0.1 else 2

    def reset(self, seed=None, return_info=False, options=None):
        self.map = np.zeros((self.sqr, self.sqr), dtype=int)
        self.generate_tile()
        self.generate_tile()
        self.gen_action_mask()
        self.step_cnt = 0
        return self.get_obs(), self._get_info()

    def rotate_map(self, action, reverse=False):
        if action == 0:
            return self.map
        if action == 1:
            return self.map.transpose((1, 0))
        if action == 2:
            return self.map[:, ::-1]
        if action == 3:
            if reverse:
                return self.map.transpose((1, 0))[::-1]
            return self.map[::-1].transpose((1, 0))

    def merge_tiles(tiles):
        new_tiles = np.zeros_like(tiles)
        point = 0
        for tile in tiles:
            if tile == 0:
                continue
            if new_tiles[point] != 0:
                if new_tiles[point] == tile:
                    new_tiles[point] += tile
                else:
                    new_tiles[point + 1] = tile
                point += 1
            else:
                new_tiles[point] = tile
        return new_tiles

    def step(self, action):
        assert self.mask[action]
        info = self._get_info()

        if not self.silent_mode:
            self.render()
            # time.sleep(0.2)

        ori_rew = np.sort(self.get_obs())[::-1]

        rot_map = self.rotate_map(action)
        self.map = np.array([M2048.merge_tiles(tiles) for tiles in rot_map], dtype=int)
        self.map = self.rotate_map(action, True)

        new_rew = np.sort(self.get_obs())[::-1]
        step_rew = new_rew - ori_rew
        step_rew[step_rew < 0] = 0

        self.generate_tile()
        self.gen_action_mask()
        self.step_cnt += 1
        if sum(self.mask) == 0:
            return self.get_obs(), -100 / np.log(self.step_cnt), True, False, info
        else:
            return self.get_obs(), sum(step_rew) + 1, False, False, info

    def drawGrid(self):
        for y in range(0, self.window_width, self.block_size):
            for x in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                num = int(self.map[int(x / self.block_size), int(y / self.block_size)])
                if num == 0:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                else:
                    color = (250, 250 - np.log(num) * 10, 250 - np.log(num) * 10)
                    pygame.draw.rect(self.screen, color, rect)
                    text = self.font.get_rect(str(num))
                    text.center = rect.center
                    self.font.render_to(self.screen, text.topleft, str(num), (0, 0, 0))
        pygame.display.update()

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), 40)
        self.screen.fill((255, 255, 255))
        self.drawGrid()

    def _get_info(self):
        return {"map": self.map}

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def gen_action_mask(self):
        self.mask = np.zeros(4, dtype=int)
        for action in range(4):
            rot_map = self.rotate_map(action)
            self.mask[action] = bool(
                (rot_map == 0).sum() or (np.diff(rot_map) == 0).sum()
            )

    def action_masks(self) -> np.ndarray[int]:
        return self.mask


if __name__ == "__main__":
    import time

    env = M2048(4, silent_mode=True)
    env.reset()

    done = False
    while not done:
        while env.mask[action := env.action_space.sample()] == 0:
            pass

        obs, reward, done, _, info = env.step(action)
        print(action)
        print(env.map)

    env.close()
