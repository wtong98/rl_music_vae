"""
Experiments with SB3 and RL-finetuning
"""

# <codecell>
import gym

# TODO: implement RNN reward
def inc_reward(history):
    if history[-1] > history[-2]:
        return 1
    return 0

class MusicEnv(gym.Env):
    def __init__(self, window_size=8, max_notes=64) -> None:
        super().__init__()

        self.num_notes = 129   # 128 MIDI notes + rest
        self.rest_idx = 128

        self.window_size = window_size
        self.max_notes = max_notes

        self.note_counter = 0
        self.note_hist = []
        self.last_obs = None

        self.observation_space = gym.spaces.MultiDiscrete(self.window_size * [self.num_notes])
        self.action_space = gym.spaces.Discrete(self.num_notes)
    
    def step(self, action):
        self.note_counter += 1
        self.note_hist.append(action)

        obs = self.last_obs[1:] + [action]
        is_done = self.note_counter == self.max_notes
        reward = inc_reward(self.note_hist)

        self.last_obs = obs
        return obs, reward, is_done, {}
    
    def reset(self):
        self.note_counter = 0
        self.note_hist = []
        obs = self.window_size * [self.rest_idx]
        self.last_obs = obs
        return obs
    

# TODO: implement (R)DQN
        