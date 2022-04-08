"""
Experiments with SB3 and RL-finetuning
"""

# <codecell>
from email import policy
import random
from collections import deque, namedtuple
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial

from tqdm import tqdm

# TODO: implement RNN reward
def inc_reward(history):
    if len(history) >=2:
        if history[-1] > history[-2]:
            return 5
    return 0

@torch.no_grad()
def note_reward(model, history, device='cpu'):
    if len(history) > 1:
        x = torch.tensor(history[:-1], device=device).unsqueeze(0)
        logits = model(x).squeeze()
        return logits[history[-1]].item()
    
    return 0


class MusicEnv(gym.Env):
    def __init__(self, note_model, window_size=32, max_notes=64, note_weight = 1, device='cpu') -> None:
        super().__init__()

        self.model = note_model
        self.device = device
        self.note_weight = note_weight

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
        inc_rew = inc_reward(self.note_hist)
        note_rew = note_reward(self.model, self.note_hist, device=self.device)
        reward = inc_rew + self.note_weight * note_rew

        self.last_obs = obs
        return obs, reward, is_done, {
            'inc_reward': inc_rew,
            'note_reward': note_rew
        }
    
    def reset(self):
        self.note_counter = 0
        self.note_hist = []
        obs = self.window_size * [self.rest_idx]
        obs[-1] = 60   # start on middle C
        self.last_obs = obs
        return obs
    

# just the decoder
class MusicDQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.emb_size = 64
        self.latent_size = 128    # paper uses 512
        self.dec_size = 512      # paper uses 1024
        self.dec_layers = 2      # paper to uses 2
        self.num_pitches = 129

        self.embedding = nn.Linear(self.num_pitches, self.emb_size)
        self.latent_to_dec = nn.Linear(self.latent_size, 2 * self.dec_size * self.dec_layers)
        self.dec = nn.LSTM(
            input_size=self.emb_size,
            batch_first=True,
            hidden_size=self.dec_size,
            num_layers=self.dec_layers,
        )

        self.dec_to_logit = nn.Linear(self.dec_size, self.num_pitches)

    def _decode(self, x, h, c):
        if x.shape[-1] != self.num_pitches:
            x = F.one_hot(x, num_classes=self.num_pitches).float()
        
        input_emb = self.embedding(x)
        dec_out, (h, c) = self.dec(input_emb, (h, c))
        logits = self.dec_to_logit(dec_out)
        return logits, h, c
    
    def forward(self, x):
        latent = torch.randn((x.shape[0], self.latent_size), device=device)  # TODO: consider more structured method
        z = self.latent_to_dec(latent)
        
        h, c = z.chunk(2, dim=-1)
        h = h.reshape(self.dec_layers, -1, self.dec_size)
        c = c.reshape(self.dec_layers, -1, self.dec_size)

        logits, _, _ = self._decode(x, h, c)
        return logits[:,-1,:]   # return last prediction

    
    def next_action(self, state, beta=1, device='cpu'):
        while state[0] == 128 and len(state) > 1:   # truncate rests
            state = state[1:]

        x = torch.tensor(state, device=device).unsqueeze(0)
        logits = self(x)[0]

        samp = Multinomial(logits=beta*logits).sample()
        return torch.argmax(samp).item()

    
    @torch.no_grad()
    def sample(self, z, max_length=32, beta=1, start_seq=None):
        z = self.latent_to_dec(z)
        h, c = z.chunk(2, dim=-1)
        h = h.reshape(self.dec_layers, -1, self.dec_size)
        c = c.reshape(self.dec_layers, -1, self.dec_size)

        all_notes = [60] if start_seq == None else start_seq
        curr_note = None

        for note in all_notes:
            note = nn.functional.one_hot(torch.tensor([[note]]), num_classes=129).float()
            preds, h, c = self._decode(note, h, c)

            probs = nn.functional.softmax(beta * preds, dim=-1).cpu().numpy()
            probs = probs / np.sum(probs)
            curr_note = np.random.choice(129, p=probs.flatten())

        curr_note = torch.tensor([[curr_note]])
        gen_out = [curr_note]
        for _ in range(max_length -1):
            note = nn.functional.one_hot(curr_note, num_classes=129).float()
            preds, h, c = self._decode(note, h, c)

            probs = nn.functional.softmax(beta * preds, dim=-1).cpu().numpy()
            probs = probs / np.sum(probs)
            curr_note = np.random.choice(129, p=probs.flatten())

            curr_note = torch.tensor([[curr_note]])
            gen_out.append(curr_note)
        
        gen_out = torch.cat(gen_out, dim=-1).squeeze(0)
        return all_notes + gen_out.tolist()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# <codecell>
state_dict = torch.load('save/model_rnn_ae.pt')
dqn_dict = {k:v.cpu() for k, v in state_dict.items() if not k.startswith('enc')}

policy_net = MusicDQN()
target_net = MusicDQN()
note_net = MusicDQN()

policy_net.load_state_dict(dqn_dict)
target_net.load_state_dict(dqn_dict)
note_net.load_state_dict(dqn_dict)


# <codecell>
batch_size = 128
gamma = 0.999
beta_start = 0
beta_end = 1
note_weight = 0
beta_decay = 200
target_update = 10
grad_clip_range = (-1, 1)
n_episodes = 300

device = 'cuda'

env = MusicEnv(note_net, note_weight=note_weight, device=device)
memory = ReplayMemory(10000)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(policy_net.parameters())

policy_net.to(device)
target_net.to(device)
note_net.to(device)

state_batch = None
def optimize_step():
    global state_batch
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor([s != None for s in batch.next_state], device=device, dtype=torch.bool)
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_vals = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).flatten()

    next_state_vals = torch.zeros(batch_size, device=device)
    next_state_vals[non_final_mask] = target_net(state_batch[non_final_mask]).max(dim=-1)[0].detach()
    exp_state_action_vals = (next_state_vals * gamma) + reward_batch
    
    loss = criterion(state_action_vals, exp_state_action_vals)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.clamp_(*grad_clip_range)
    optimizer.step()


# TODO: gather reward statistics, try arp strat
total_reward = 0
inc_rew = 0
note_rew = 0
steps = 0

for e in tqdm(range(n_episodes)):
    obs = env.reset()
    is_done = False

    while not is_done:
        beta = beta_end + (beta_start - beta_end) * np.exp(-(steps / beta_decay))
        action = policy_net.next_action(obs, beta=beta, device=device)   # TODO: beta scheduling
        next_obs, reward, is_done, info = env.step(action)
        memory.push(
            torch.tensor(obs, device=device),
            torch.tensor(action, device=device),
            torch.tensor(next_obs, device=device),
            torch.tensor(reward, device=device))
        
        obs = next_obs
        optimize_step()

        total_reward += reward
        inc_rew += info['inc_reward']
        note_rew += info['note_reward']
        steps += 1
    
    if e % target_update == 0 and e > 0:
        avg_reward = total_reward / target_update
        print(f'Episode {e}:  reward={avg_reward:.2f}   inc={inc_rew / target_update:.2f}  note={note_rew / target_update:.2f}')
        target_net.load_state_dict(policy_net.state_dict())
        total_reward = 0
        inc_rew = 0
        note_rew = 0


# %%
policy_net.cpu()
z = torch.randn((1,128))
seq = policy_net.sample(z, beta=1)
seq

# %%
