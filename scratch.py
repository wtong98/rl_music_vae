"""
Some experimentation with the data and music models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from music21 import corpus
from music21 import note as nt
from music21.converter import parse
from music21 import stream

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from tqdm import tqdm

REST_IDX = 128

def batch(score, sampling_rate=0.25, bars=2):
    text = _to_text(score, sampling_rate=sampling_rate)
    text = np.concatenate(text, axis=0)

    # samps_per_batch = int(4 / sampling_rate * bars)
    samps_per_batch = 32
    total_samps = text.shape[0] // samps_per_batch

    batch = np.zeros((total_samps, samps_per_batch, 129))
    for i in range(total_samps):
        start_idx = i * samps_per_batch
        end_idx = (i + 1) * samps_per_batch
        # print('BATCH', batch[i,:,:].shape)
        # print('TEXT', text[start_idx:end_idx,:].shape)
        batch[i,:,:] = text[start_idx:end_idx,:]

    return batch.astype('float32')


def _to_text(score, sampling_rate) -> list:
    notes = score.flat.getElementsByClass(nt.Note)
    hist = _bin(notes, sampling_rate)
    end = score.flat.highestOffset

    text = [_to_word(hist[i]) for i in np.arange(0, end, sampling_rate)]
    return text

def _bin(notes, sampling_rate) -> defaultdict:
    hist = defaultdict(list)

    for note in notes:
        offset = note.offset
        halt = offset + note.duration.quarterLength

        if _precise_round(offset % sampling_rate) != 0:
            offset = _precise_round(offset - (offset % sampling_rate))
        if _precise_round(halt % sampling_rate) != 0:
            halt = _precise_round(halt + (sampling_rate - halt % sampling_rate))

        while offset < halt:
            hist[offset].append(note)
            offset += sampling_rate

    return hist

def _to_word(notes) -> str:
    if len(notes) == 0:
        one_hot = np.zeros((1, 129))
        one_hot[0, REST_IDX] = 1
        return one_hot

    # arp notes that occur together
    # TODO: might be interesting to arp across parts
    ordered_notes = sorted(notes, key=lambda n: n.pitch.midi, reverse=False)
    one_hot = np.zeros((len(ordered_notes), 129))
    for i, note in enumerate(ordered_notes):
        one_hot[i, note.pitch.midi] = 1

    return one_hot

def _precise_round(val, precision=10):
    return round(val * precision) / precision


# <codecell>
bundle = corpus.search('bach', 'composer')
scores = [metadata.parse() for metadata in tqdm(bundle)]

# %%
# examples = [batch(score, sampling_rate=0.5) for score in scores]
examples = [batch(part) for score in scores for part in score.parts]
examples = np.concatenate(examples, axis=0)

# %%
dataset = TensorDataset(torch.tensor(examples))

test_len = int(examples.shape[0] * 0.1)
train_len = examples.shape[0] - test_len
train_ds, test_ds = random_split(dataset, (train_len, test_len))

# <codecell>
class MusicVAE(nn.Module):
    def __init__(self, num_bars=2) -> None:
        super().__init__()

        self.num_bars = num_bars
        self.num_samples = num_bars * 16
        self.num_pitches = 129

        self.emb_size = 64

        self.enc_size = 512      # paper uses 2048
        self.enc_layers = 2      # paper uses 2
        self.latent_size = 128    # paper uses 512

        self.cond_size = 256     # paper uses 1024
        self.cond_layers = 2
        self.cond_out = 128       # paper uses 512  # TODO: need projection layers to match

        self.dec_size = 256      # paper uses 1024
        self.dec_layers = 1      # paper to uses 2
        self.dec_out = 128        # paper uses 512


        # self.embedding = nn.Embedding(self.num_pitches, self.emb_size)
        self.embedding = nn.Linear(self.num_pitches, self.emb_size)

        self.encoder = nn.LSTM(
            input_size=self.emb_size,
            batch_first=True,
            hidden_size=self.enc_size,
            num_layers=self.enc_layers,
            bidirectional=True
        )

        enc_out_dim = 2 * self.enc_size * self.num_samples
        self.enc_lin_1 = nn.Linear(enc_out_dim, 2 * self.latent_size)  # paper uses 2 layers her

        self.latent_to_cond = nn.Linear(self.latent_size, 2 * self.cond_out)
        self.cond = nn.LSTM(
            input_size=self.cond_size,
            batch_first=True,
            hidden_size=self.cond_size,
            num_layers=self.cond_layers,
            proj_size=self.cond_out
        )

        self.cond_to_dec = nn.Linear(self.cond_out, self.dec_out)
        self.dec = nn.LSTM(
            input_size=self.dec_out + self.emb_size,
            batch_first=True,
            hidden_size=self.dec_size,
            num_layers=self.dec_layers,
            proj_size=self.dec_out
        )

        self.dec_to_logit = nn.Linear(self.dec_out, self.num_pitches)
    
    @torch.no_grad()
    def _reparam(self, mu, sig):
        batch_size = mu.size(0)
        eps = torch.randn(batch_size, self.latent_size, device='cuda')
        z = eps * torch.log(torch.exp(sig) + 1) + mu
        return z

    def _encode(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.enc_lin_1(x)

        mu, sig = torch.chunk(x, 2, dim=1)
        return mu, sig
    
    def _decode(self, z):
        batch_size = z.shape[0]
        z = self.latent_to_cond(z)
        z = nn.functional.tanh(z)
        z = torch.reshape(z, (2, batch_size, self.cond_out))

        cond_in = torch.zeros(batch_size, self.num_bars, self.cond_size, device='cuda')
        cond_cell = torch.zeros((2, batch_size, self.cond_size), device='cuda')
        dec_h, _ = self.cond(cond_in, (z, cond_cell))

        all_logits = []
        all_one_hots = []
        dec_cell = torch.zeros((1, batch_size, self.dec_size), device='cuda')

        for i in range(self.num_bars):
            dec_z = dec_h[:,i,:]
            dec_z = self.cond_to_dec(dec_z)
            dec_z = torch.reshape(dec_z, (1, batch_size, self.dec_out))

            if len(all_one_hots) > 0:
                last_one_hot = all_one_hots[-1]
            else:
                last_one_hot = torch.zeros(batch_size, self.emb_size, device='cuda')

            dec_in = torch.concat((dec_z.squeeze(), last_one_hot), dim=-1)
            dec_in = dec_in.unsqueeze(1)

            for _ in range(16):
                out, (dec_z, dec_cell) = self.dec(dec_in, (dec_z, dec_cell))

                logit = self.dec_to_logit(out[:,0,:])
                all_logits.append(logit)

                one_hot_idx = torch.argmax(logit, dim=1)
                one_hots = nn.functional.one_hot(one_hot_idx, num_classes=self.num_pitches).float()
                one_hots = self.embedding(one_hots)
                dec_in = torch.concat((dec_in[:,0,:self.dec_out], one_hots), dim=-1)
                dec_in = dec_in.unsqueeze(1)
                
                all_one_hots.append(one_hots)

        out = torch.stack(all_logits, dim=1)
        return out

    def forward(self, x):
        mu, sig = self._encode(x)
        z = self._reparam(mu, sig)
        logits = self._decode(z)

        return logits
    
    def loss(self, x, x_reco, kl_weight=0.5):
        targets = torch.argmax(x, dim=-1)
        targets = targets.reshape(-1)
        logits = x_reco.reshape(-1, self.num_pitches)
        base_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')

        mu, sig = self._encode(x)
        sig = torch.log(torch.exp(sig) + 1)

        prior = Normal(0, 1)
        pred = Normal(mu, sig)
        kl_loss = kl_divergence(pred, prior).mean()

        total = base_loss + kl_weight * kl_loss

        return {
            'ce': base_loss,
            'kl': kl_loss,
            'mu': mu,
            'sig': sig,
            'total': total
        }

# <codecell>
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, pin_memory=True)

model = MusicVAE()
opt = Adam(model.parameters())

model.cuda()

train_losses = []
test_losses = []
num_epochs = 20

# iters = 0
# eval_every = 1000

for epoch in range(num_epochs):

    with torch.no_grad():
        model.eval()
        curr_loss = []
        kl_loss = []
        means = []
        sigs = []
        for x in tqdm(test_loader):
            x = x[0].cuda()
            x_reco = model(x)
            loss = model.loss(x, x_reco)
            curr_loss.append(loss['total'].item())
            kl_loss.append(loss['kl'].item())
            means.append(np.mean(loss['mu'].cpu().numpy()))
            sigs.append(np.mean(loss['sig'].cpu().numpy()))

        test_losses.append(np.mean(curr_loss))
        model.train()
        print('Eval loss:', np.mean(curr_loss))
        print('KL: ', np.mean(kl_loss))
        print('Mu:', np.mean(means))
        print('Sig:', np.mean(sigs))

    for x in tqdm(train_loader):
        x = x[0].cuda()

        opt.zero_grad()
        x_reco = model(x)
        loss = model.loss(x, x_reco)

        loss['total'].backward()
        opt.step()

        train_losses.append(loss['total'].item())

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
plt.savefig('fig/loss.png')

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
    score.write('midi', fp=f'sample/{i}.mid')

# %%
