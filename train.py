"""
Some experimentation with the data and music models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt

from music21 import note as nt
from music21 import stream

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from data import load_composer, scores_to_dataset
from model import MusicVAE

# <codecell>
scores = load_composer(name='bach')
dataset = scores_to_dataset(scores)

test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_ds, test_ds = random_split(dataset, (train_len, test_len))

# <codecell>
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, pin_memory=True)

@torch.no_grad()
def evaluate_model(model, test_dl):
    total_loss = 0
    num_iters = 0

    curr_loss = []
    kl_loss = []
    means = []
    sigs = []
    for x in tqdm(test_loader):
        x = x[0].cuda()
        x_reco = model(x)
        loss = model.loss(x, x_reco)
        total_loss += loss['total'].item()

        kl_loss.append(loss['kl'].item())
        means.append(np.mean(loss['mu'].cpu().numpy()))
        sigs.append(np.mean(loss['sig'].cpu().numpy()))
        num_iters += 1

    print('Eval loss:', np.mean(curr_loss))
    print('KL: ', np.mean(kl_loss))
    print('Mu:', np.mean(means))
    print('Sig:', np.mean(sigs))
    return total_loss / num_iters

model = MusicVAE()
opt = Adam(model.parameters())

model.cuda()

train_losses = []
test_losses = []
num_epochs = 20

# iters = 0
# eval_every = 1000

for epoch in range(num_epochs):
    total_loss = 0
    num_iters = 0

    for x in tqdm(train_loader):
        x = x[0].cuda()

        opt.zero_grad()
        x_reco = model(x)
        loss = model.loss(x, x_reco)

        loss['total'].backward()
        opt.step()

        total_loss += loss['total'].item()
        num_iters += 1
    
    train_losses.append(total_loss / num_iters)

    model.eval()
    test_loss = evaluate_model(model, test_loader)
    model.train()
    test_losses.append(test_loss)

with torch.no_grad():
    model.eval()
    curr_loss = []
    kl_loss = []
    for x in tqdm(test_loader):
        x = x[0].cuda()
        x_reco = model(x)
        loss = model.loss(x, x_reco)
        curr_loss.append(loss['total'].item())
        kl_loss.append(loss['kl'].item())

    test_losses.append(np.mean(curr_loss))
    print('Eval loss:', np.mean(curr_loss))
    print('KL: ', np.mean(kl_loss))

# <codecell>
plt.plot(np.arange(num_epochs), train_losses[::2*134][:-1], '--o', label='Train loss')
plt.plot(np.arange(num_epochs), test_losses[:-1], '--o', label='Test loss')
plt.xticks(np.arange(num_epochs)[::2])
plt.legend()

plt.title('MusicVAE Loss')
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.savefig('save/fig/loss.png')

# <codecell>
## TODO: save model

# <codecell>
with torch.no_grad():
    samp = torch.randn((10, 128)).cuda()
    logits = model._decode(samp).cpu()

# <codecell>
def logits_to_score(logits, beta=5):
    all_scores = []
    for note_set in logits:
        score = stream.Stream()
        for note in note_set:
            probs = np.exp(beta * note) / np.sum(np.exp(beta * note))
            midi_val = np.random.choice(129, p=probs.flatten())
            note = nt.Note(midi_val)
            score.append(note)
        
        all_scores.append(score)
    
    return all_scores

all_scores = logits_to_score(logits.numpy())
        
# %%
for i, score in enumerate(all_scores):
    score.write('midi', fp=f'save/sample/{i}.mid')

# %%
