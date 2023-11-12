from M2048 import M2048
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


# env = ActionMasker(env, MinesweeperEnv.get_action_mask)
# env = DummyVecEnv([lambda: env])
model = MaskablePPO.load("./model/2048.pkl")

for i in range(10):
    env = M2048(4, silent_mode=False)
    state, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(observation=state)
        while not env.action_masks()[action] and env.action_masks().sum():
            action, _ = model.predict(observation=state)
            print(action)

        state, reward, done, _, info = env.step(action)
        score += reward
    print(env.map)

env.close()
print(score)
