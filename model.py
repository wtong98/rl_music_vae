"""
Model definitions

author: William Tong (wtong@g.harvard.edu)
"""

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