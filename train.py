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
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.distributions import Normal, kl_divergence

from tqdm import tqdm

from data import load_all, load_composer, scores_to_dataset
from model import MusicVAE, MusicAE, RnnAE

# <codecell>
# scores = load_composer(name='bach')
# dataset = scores_to_dataset(scores, sampling_rate=0.5)
all_ds = load_all()
dataset = ConcatDataset(all_ds.values())

test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_ds, test_ds = random_split(dataset, (train_len, test_len))

# <codecell>
class RnnVAE(nn.Module):
    def __init__(self, num_bars=2) -> None:
        super().__init__()

        self.num_bars = num_bars
        self.num_samples = num_bars * 16
        self.num_pitches = 129

        self.emb_size = 64

        self.enc_size = 512      # paper uses 2048
        self.enc_layers = 2      # paper uses 2

        self.latent_size = 128    # paper uses 512

        self.dec_size = 512      # paper uses 1024
        self.dec_layers = 2      # paper to uses 2


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
        self.enc_lin_1 = nn.Linear(enc_out_dim, 2 * self.latent_size)  # paper uses 2 layers here

        self.latent_to_dec = nn.Linear(self.latent_size, 2 * self.dec_size * self.dec_layers)
        self.dec = nn.LSTM(
            input_size=self.emb_size,
            batch_first=True,
            hidden_size=self.dec_size,
            num_layers=self.dec_layers,
        )

        self.dec_to_logit = nn.Linear(self.dec_size, self.num_pitches)
    
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

        mu, sig = torch.chunk(x, 2, dim=-1)
        return mu, sig

    def _decode(self, x, h, c):
        input_emb = self.embedding(x)
        all_logits = []

        for i in range(x.shape[1]):
            if np.random.random() > 0.5 or len(all_logits) == 0:
                input_tok = input_emb[:,i,:].unsqueeze(1)
            else:
                input_tok = torch.argmax(all_logits[-1], dim=-1)
                input_tok = torch.nn.functional.one_hot(input_tok, num_classes=self.num_pitches).float()
                input_tok = self.embedding(input_tok)

            dec_out, (h, c) = self.dec(input_tok, (h, c))
            logits = self.dec_to_logit(dec_out)
            all_logits.append(logits)

        out = torch.cat(all_logits, dim=1)
        return out, h, c

    def forward(self, x):
        mu, sig = self._encode(x)
        z = self._reparam(mu, sig)
        z = self.latent_to_dec(z)
        
        # z = torch.zeros(z.shape).cuda()  # TODO: need more diverse dataset to prevent raw memorizing
        h, c = z.chunk(2, dim=-1)
        h = h.reshape(self.dec_layers, -1, self.dec_size)
        c = c.reshape(self.dec_layers, -1, self.dec_size)

        logits, _, _ = self._decode(x, h, c)
        return logits
    
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
            
    # TODO: fix loss
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
    return total_loss / num_iters, (total_acc / num_iters).item()

model = RnnVAE()
opt = Adam(model.parameters())

model.cuda()

train_losses = []
test_losses = []
test_accs = []
num_epochs = 10

for epoch in range(num_epochs):
    print('EPOCH:', epoch+1)
    total_loss = 0
    num_iters = 0

    model.eval()
    test_loss, test_acc = evaluate_model(model, test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
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
test_loss, test_acc = evaluate_model(model, test_loader)
model.train()
test_losses.append(test_loss)
test_accs.append(test_acc)

# <codecell>
plt.plot(np.arange(num_epochs), train_losses, '--o', label='Train loss')
plt.plot(np.arange(num_epochs), test_losses[:-1], '--o', label='Test loss')
plt.xticks(np.arange(num_epochs)[::2])
plt.legend(loc='center right')
plt.title('RNN VAE Training Performance')
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')

ax = plt.gca().twinx()
ax.plot(np.arange(num_epochs), test_accs[:-1], '--o', label='Test accuracy', color='tab:orange')
ax.set_ylabel('Test accuracy', color='tab:orange')
ax.tick_params(axis='y', labelcolor='tab:orange')


plt.savefig('save/fig/loss.png')

# <codecell>
torch.save(model.state_dict(), 'save/model_rnn_vae_all.pt')

# <codecell>
model = RnnVAE()
state_dict = torch.load('save/model_rnn_vae_all.pt')
model.load_state_dict(state_dict)
model.eval()

# <codecell>
model.cpu()
# <codecell>
# ex = next(iter(test_loader))[0]
# print(torch.argmax(ex[0], dim=-1))
# mu, sig = model._encode(ex)
z = torch.randn((1, 128))
# z = model._reparam(mu, sig)

# <codecell>
N = 5

for i in range(N):
    samp = model.sample(z[:1, :], start_seq=[60], beta=0.8)
    print('samp', samp)
    note_dur = 0.5
    last_note = None
    all_notes = []
    score = stream.Stream()
    for note in samp:
        if note == 128:
            elem = nt.Rest()
        else:
            if note == last_note:
                all_notes[-1].quarterLength += note_dur
            else:
                elem = nt.Note(note)
                elem.quarterLength = 0.5
                last_note = note
                all_notes.append(elem)

    for n in all_notes:
        score.append(n)

    score.write('midi', f'save/sample/{i}.mid')
        

# %%
### PRODUCE SAMPLES OF EACH 'STYLE'
model.cuda()

n_examples = 128
all_vecs = {}

for name, ds in all_ds.items():
    print('processing', name)
    spare = len(ds) - n_examples
    sample, _ = random_split(ds, [n_examples, spare])
    all_examples = next(iter(DataLoader(ds, batch_size=n_examples)))[0]
    all_examples = all_examples.cuda()

    all_out, sig = model._encode(all_examples)
    platonic_rep = torch.mean(all_out, axis=0)
    all_vecs[name] = platonic_rep

# %%
model.cpu()

for name, z in all_vecs.items():
    z = z.cpu()
    z = z.unsqueeze(0)
    note_dur = 0.5
    last_note = None

    samp = model.sample(z, start_seq=[60], beta=1)
    print('samp', samp)
    all_notes = []
    score = stream.Stream()
    for note in samp:
        if note == 128:
            elem = nt.Rest()
        else:
            if note == last_note:
                all_notes[-1].quarterLength += note_dur
            else:
                elem = nt.Note(note)
                elem.quarterLength = 0.5
                last_note = note
                all_notes.append(elem)

    for n in all_notes:
        score.append(n)
    
    score.write('midi', f'save/sample/{name}.mid')
    



# %%
