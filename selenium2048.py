import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
import re
import numpy as np
from selenium.webdriver.common.keys import Keys


class Selenium2048(gym.Env):
    def __init__(self):
        ch_options = webdriver.ChromeOptions()

        # 为Chrome配置无头模式
        # ch_options.add_argument("--headless")
        # ch_options.add_argument('--no-sandbox')
        # ch_options.add_argument('--disable-gpu')
        # ch_options.add_argument('--disable-dev-shm-usage')
        ch_options.add_argument("--mute-audio")

        self.dr = webdriver.Chrome(options=ch_options)
        self.dr.set_window_size(800, 1200)

        url = "http://www.2048123.com/"
        self.dr.get(url)

        self.observation_space = spaces.Box(
            0, 2 << 15, shape=self.get_obs().shape, dtype=int
        )
        self.action_space = spaces.Discrete(4)
        self.retry_btn = self.dr.find_element(by=By.CLASS_NAME, value="retry-button")

    def get_obs(self):
        tiles = self.dr.find_element(
            by=By.CLASS_NAME, value=r"tile-container"
        ).find_elements(by=By.XPATH, value="./div")

        map = np.zeros((4, 4))
        pattern = r"\b\d+\b"
        for tile in tiles:
            class_name = tile.get_attribute("class")
            matches = re.findall(pattern, class_name)
            map[int(matches[2]) - 1, int(matches[1]) - 1] = int(matches[0])
        self.map = map
        return map.flatten()

    def reset(self, seed=None, return_info=False, options=None):
        res_btn = self.dr.find_element(by=By.CLASS_NAME, value="restart-button")
        res_btn.click()
        self.step_cnt = 0
        return self.get_obs(), self._get_info()

    def step(self, action):
        info = self._get_info()
        ori_rew = np.sort(self.get_obs())[::-1]

        body_element = self.dr.find_element(by=By.TAG_NAME, value="body")
        if action == 0:
            body_element.send_keys(Keys.ARROW_LEFT)
        if action == 1:
            body_element.send_keys(Keys.ARROW_UP)
        if action == 2:
            body_element.send_keys(Keys.ARROW_RIGHT)
        if action == 3:
            body_element.send_keys(Keys.ARROW_DOWN)

        new_rew = np.sort(self.get_obs())[::-1]
        step_rew = new_rew - ori_rew
        step_rew[step_rew < 0] = 0

        self.step_cnt += 1

        if self.retry_btn.is_displayed():
            return self.get_obs(), -100 / np.log(self.step_cnt), True, False, info
        else:
            return (
                self.get_obs(),
                sum(step_rew) + np.log(self.step_cnt),
                False,
                False,
                info,
            )

    def _get_info(self):
        return {"map": self.map}

    def close(self):
        self.dr.close()


if __name__ == "__main__":
    import time

    en2 = Selenium2048()
    en2.reset()

    done = False
    while not done:
        action = en2.action_space.sample()
        obs, reward, done, _, info = en2.step(action)
        print(action)
        time.sleep(0.5)

    en2.close()
