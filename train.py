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
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from scipy.special import logsumexp
from tqdm import tqdm

from data import load_composer, scores_to_dataset
from model import MusicVAE, MusicAE, RnnAE

# <codecell>
scores = load_composer(name='bach')
dataset = scores_to_dataset(scores, sampling_rate=0.5)

test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_ds, test_ds = random_split(dataset, (train_len, test_len))

# <codecell>
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, pin_memory=True)

@torch.no_grad()
def logits_to_idx(logits, beta=1):
    vals_set = []
    for note_set in logits:
        vals = []
        for note in note_set:
            log_probs = beta * note - torch.logsumexp(beta * note, dim=0)
            log_probs = log_probs.cpu().numpy()
            midi_val = np.random.choice(129, p=np.exp(log_probs).flatten())
            vals.append(midi_val)
        vals_set.append(vals)
    
    vals_set = torch.tensor(vals_set)
    return vals_set

@torch.no_grad()
def evaluate_model(model, test_dl):
    total_loss = 0
    num_iters = 0
    total_acc = 0

    kl_loss = []
    means = []
    sigs = []
    for notes in tqdm(test_dl):
        x = notes[0].cuda()
        x_reco = model(x)
        loss = model.loss(x, x_reco)
        total_loss += loss['total'].item()

        preds = logits_to_idx(x_reco)
        total_acc += torch.mean((torch.argmax(notes[0], dim=-1) == preds).float())

        kl_loss.append(loss['kl'].item())
        means.append(np.mean(loss['mu'].cpu().numpy()))
        sigs.append(np.mean(loss['sig'].cpu().numpy()))
        num_iters += 1

    print('Eval loss:', total_loss / num_iters)
    print('KL: ', np.mean(kl_loss))
    print('Mu:', np.mean(means))
    print('Sig:', np.mean(sigs))
    print('Acc:', total_acc / num_iters)
    return total_loss / num_iters

model = RnnAE()
opt = Adam(model.parameters())

model.cuda()

train_losses = []
test_losses = []
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    num_iters = 0

    model.eval()
    test_loss = evaluate_model(model, test_loader)
    model.train()

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

# <codecell>
plt.plot(np.arange(num_epochs), train_losses, '--o', label='Train loss')
plt.plot(np.arange(num_epochs), test_losses, '--o', label='Test loss')
plt.xticks(np.arange(num_epochs)[::2])
plt.legend()

plt.title('MusicVAE Loss')
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.savefig('save/fig/loss.png')

# <codecell>
## TODO: save model
torch.save(model.state_dict(), 'save/model_rnn_ae.pt')

# <codecell>
state_dict = torch.load('save/model_rnn_ae.pt')
model.load_state_dict(state_dict)
model.eval()

# <codecell>
model.cpu()
# <codecell>
ex = next(iter(test_loader))[0]
print(torch.argmax(ex[0], dim=-1))
z = model._encode(ex)

# <codecell>
N = 5
all_scores = []

for _ in range(N):
    samp = model.sample(z[:1, :], start_seq=[60], beta=0.5)
    score = stream.Stream()
    for note in samp:
        elem = nt.Note(note)
        score.append(elem)
    all_scores.append(score)

        
# %%
for i, score in enumerate(all_scores):
    score.write('midi', fp=f'save/sample/{i}.mid')

# %%
