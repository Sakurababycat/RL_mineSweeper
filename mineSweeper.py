import gymnasium as gym
from gymnasium import spaces
import numpy as np
from random import randrange
import random
import pygame
import pygame.freetype
import time
import torch


class MinesweeperEnv(gym.Env):
    # a minesweeper environment implemented in openai gym style
    # STATE
    #   given the map of height x width, numbers in each block of map:
    #   1. -1 denotes unopened block
    #   2. 0-8 denotes the number of mines in the surrounding 8 blocks of the block
    # MAP
    #   the map is unobervable to the agent where 0 is non-mine, 1 has a mine
    # ACTIONS
    #   action is Discrete(height*width) representing an attempt to open at the block's
    #   index when the map is flattened.
    # RESET
    #   be sure to call the reset function before using any other function in the class
    # STEP(ACTION)
    #   returns a four tuple: next_state, reward, done, _
    # RENDER
    #   renders the current state using pygame

    def __init__(self, height=16, width=16, num_mines=40, seed=None, silent_mode=True):
        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.win_reward = 500
        self.fail_reward = -100
        self.map = np.array([[False] * width for _ in range(height)])
        self.state = np.zeros((height, width), dtype=int) - 1
        self.step_cntr = 0
        self.step_cntr_max = (height * width - num_mines) * 2

        self.block_size = 25
        self.window_height = self.block_size * height
        self.window_width = self.block_size * width
        self.map = None
        self.generate_mines()

        self.screen = None
        self.issuccess = False
        if not seed:
            seed = time.time()
        random.seed(seed)
        self.silent_mode = silent_mode

        self.observation_space = spaces.Box(
            -1, 8, shape=self.get_obs().shape, dtype=int
        )
        self.action_space = spaces.Discrete(height * width)

    def get_obs(self):
        return self.state.transpose((1, 0)).flatten()

    def generate_mines(self):
        self.map = np.array([[False] * self.width for _ in range(self.height)])
        for _ in range(self.num_mines):
            x = randrange(self.height)
            y = randrange(self.width)
            while self.map[x, y]:
                x = randrange(self.height)
                y = randrange(self.width)
            self.map[x, y] = True

    def reset(self, seed=None, return_info=False, options=None):
        self.generate_mines()
        self.step_cntr = 0
        self.issuccess = False
        self.state = np.zeros((self.height, self.width), dtype=int) - 1
        return self.get_obs(), self._get_info()

    def get_num_opened(self):
        count = 0
        for i in self.state.flatten():
            if i >= 0:
                count += 1
        return count

    def get_num_surr(self, x, y):
        count = 0
        for i in range(max(0, x - 1), min(self.height, x + 2)):
            for j in range(max(0, y - 1), min(self.width, y + 2)):
                if not (i == x and j == y):
                    if self.map[i, j]:
                        count += 1
        return count

    def update_state(self, x, y):
        num_surr = self.get_num_surr(x, y)
        self.state[x, y] = num_surr
        if num_surr == 0:
            for i in range(max(0, x - 1), min(self.height, x + 2)):
                for j in range(max(0, y - 1), min(self.width, y + 2)):
                    if (not (i == x and j == y)) and self.state[i, j] == -1:
                        self.update_state(i, j)

    def step(self, action):
        action = [action % self.width, action // self.width]
        if (
            len(action) != 2
            or action[0] < 0
            or action[0] >= self.height
            or action[1] < 0
            or action[1] >= self.width
        ):
            raise ValueError
        info = self._get_info()
        if self.step_cntr == self.step_cntr_max:
            return self.get_obs(), 0, True, False, info
        else:
            self.step_cntr += 1
        x, y = action[0], action[1]
        if self.map[x][y]:
            return self.get_obs(), self.fail_reward, True, False, info
        else:
            num_opened = self.get_num_opened()
            if self.state[x, y] != -1:
                return self.get_obs(), -1, False, False, info
            self.update_state(x, y)
            if not self.silent_mode:
                self.render()
                time.sleep(0.2)
            new_num_opened = self.get_num_opened()
            if new_num_opened == self.height * self.width - self.num_mines:
                self.issuccess = True
                info = self._get_info()
                return self.get_obs(), self.win_reward, True, False, info

            rew_rate = self.step_cntr * (
                1 + new_num_opened / (self.height * self.width - self.num_mines)
            )
            return (
                self.get_obs(),
                (new_num_opened - num_opened) * rew_rate,
                False,
                False,
                info,
            )

    def drawGrid(self):
        for y in range(0, self.window_width, self.block_size):
            for x in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                num = int(
                    self.state[int(x / self.block_size), int(y / self.block_size)]
                )
                if num == -1:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
                else:
                    color = (250, 250 - num * 30, 250 - num * 30)
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
            self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), 13)
        self.screen.fill((0, 0, 0))
        self.drawGrid()

    def _get_info(self):
        return {"map": self.map, "is_success": self.issuccess}

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def action_masks(self) -> np.ndarray[int]:
        state = self.get_obs().reshape(self.height, self.width)
        pad = torch.constant_pad_nd(torch.from_numpy(state).int(), (1, 1, 1, 1), -1)
        mask = (
            torch.conv2d(pad.unsqueeze(0), torch.ones((1, 1, 3, 3), dtype=torch.int))
            != -9
        ) * (state == -1)
        return mask.flatten()


if __name__ == "__main__":
    import time

    env = MinesweeperEnv(9, 9, 10, silent_mode=False)
    env.reset()
    for i in range(50):
        action = i
        env.step(action)
        print(env.action_masks().reshape(9, 9))
    env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        time.sleep(0.5)

    env.close()
