"""
Model definitions

author: William Tong (wtong@g.harvard.edu)
"""

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

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


class MusicAE(nn.Module):
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
        self.enc_lin_1 = nn.Linear(enc_out_dim, self.latent_size)  # paper uses 2 layers here

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
    
    def _encode(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.enc_lin_1(x)
        return x

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
        z = self._encode(x)
        logits = self._decode(z)

        return logits
    
    def loss(self, x, x_reco, kl_weight=0.5):
        targets = torch.argmax(x, dim=-1)
        targets = targets.flatten()
        logits = x_reco.flatten(end_dim=-2)
        base_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')

        return {
            'ce': base_loss,
            'kl': torch.tensor(0),
            'mu': torch.tensor(0),
            'sig': torch.tensor(0),
            'total': base_loss
        }


class RnnAE(nn.Module):
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
        self.enc_lin_1 = nn.Linear(enc_out_dim, self.latent_size)  # paper uses 2 layers here

        self.latent_to_dec = nn.Linear(self.latent_size, 2 * self.dec_size * self.dec_layers)
        self.dec = nn.LSTM(
            input_size=self.emb_size,
            batch_first=True,
            hidden_size=self.dec_size,
            num_layers=self.dec_layers,
        )

        self.dec_to_logit = nn.Linear(self.dec_size, self.num_pitches)
    
    def _encode(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.enc_lin_1(x)
        return x

    def _decode(self, x, h, c):
        input_emb = self.embedding(x)
        dec_out, (h, c) = self.dec(input_emb, (h, c))
        logits = self.dec_to_logit(dec_out)
        return logits, h, c

    def forward(self, x):
        z = self._encode(x)
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
        targets = targets.flatten()
        logits = x_reco.flatten(end_dim=-2)
        base_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')

        return {
            'ce': base_loss,
            'kl': torch.tensor(-1),
            'mu': torch.tensor(0),
            'sig': torch.tensor(0),
            'total': base_loss
        }


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