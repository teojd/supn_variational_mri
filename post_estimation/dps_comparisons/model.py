import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 

def timestep_embedding(timesteps, dim, max_period=0.005):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Sequential(nn.SiLU(), nn.Linear(input_dim, output_dim))
    def forward(self, x):
        return self.dense(x)[..., None, None]

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x.permute(1, 2, 0).view(*x.shape[1:], int(np.sqrt(x.shape[0])), -1)  # (B, C, H, W)

def timestep_embedding(timesteps, dim, max_period=0.005):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Sequential(nn.SiLU(), nn.Linear(input_dim, output_dim))
    def forward(self, x):
        return self.dense(x)[..., None, None]

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x.permute(1, 2, 0).view(*x.shape[1:], int(np.sqrt(x.shape[0])), -1)  # (B, C, H, W)

class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[16,32,64,128,256,512], embed_dim=256, conditions=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.conditions = conditions
        self.emb_layers = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))

        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(8, num_channels=channels[1])
        self.transformer2 = TransformerBlock(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.transformer3 = TransformerBlock(channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.transformer4 = TransformerBlock(channels[3])
        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[4])
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4])
        self.transformer5 = TransformerBlock(channels[4])
        self.conv6 = nn.Conv2d(channels[4], channels[5], 3, stride=2, padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[5])
        self.gnorm6 = nn.GroupNorm(32, num_channels=channels[5])
        self.transformer6 = TransformerBlock(channels[5])

        self.tconv6 = nn.ConvTranspose2d(channels[5], channels[4], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense7 = Dense(embed_dim, channels[4])
        self.tgnorm6 = nn.GroupNorm(32, num_channels=channels[4])
        self.transformer7 = TransformerBlock(channels[4])
        self.tconv5 = nn.ConvTranspose2d(channels[4] * 2, channels[3], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense8 = Dense(embed_dim, channels[3])
        self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])
        self.transformer8 = TransformerBlock(channels[3])
        self.tconv4 = nn.ConvTranspose2d(channels[3] * 2, channels[2], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense9 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.transformer9 = TransformerBlock(channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense10 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(8, num_channels=channels[1])
        self.transformer10 = TransformerBlock(channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense11 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, 2, 3, stride=1, padding=1)
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        return self.forward_unscaled(x, t)/self.marginal_prob_std(t)[:, None, None, None]

    def forward_unscaled(self, x, t):
        embed = self.emb_layers(timestep_embedding(t, self.embed_dim))
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h2 = self.transformer2(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h3 = self.transformer3(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        h4 = self.transformer4(h4)
        h5 = self.conv5(h4)
        h5 += self.dense5(embed)
        h5 = self.gnorm5(h5)
        h5 = self.act(h5)
        h5 = self.transformer5(h5)
        h6 = self.conv6(h5)
        h6 += self.dense6(embed)
        h6 = self.gnorm6(h6)
        h6 = self.act(h6)
        h6 = self.transformer6(h6)

        h = self.tconv6(h6)
        h += self.dense7(embed)
        h = self.tgnorm6(h)
        h = self.act(h)
        h = self.transformer7(h)
        h = self.tconv5(torch.cat([h, h5], dim=1))
        h += self.dense8(embed)
        h = self.tgnorm5(h)
        h = self.act(h)
        h = self.transformer8(h)
        h = self.tconv4(torch.cat([h, h4], dim=1))
        h += self.dense9(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.transformer9(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense10(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.transformer10(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense11(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h
        return h

class ScoreNet6(nn.Module):
    def __init__(self, marginal_prob_std, channels=[4,8,16,32,64,128], embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb_layers = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))

        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(8, num_channels=channels[1])
        self.transformer2 = TransformerBlock(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(16, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[4])
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4])

        self.transformer5 = TransformerBlock(channels[4])

        self.tconv5 = nn.ConvTranspose2d(channels[4], channels[3], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense8 = Dense(embed_dim, channels[3])
        self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])
        self.tconv4 = nn.ConvTranspose2d(channels[3] * 2, channels[2], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense9 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(16, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense10 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(8, num_channels=channels[1])
        self.transformer10 = TransformerBlock(channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, bias=False, output_padding=1, padding=1)
        self.dense11 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, 2, 3, stride=1, padding=1)
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        return self.forward_unscaled(x, t)/self.marginal_prob_std(t)[:, None, None, None]

    def forward_unscaled(self, x, t):
        embed = self.emb_layers(timestep_embedding(t, self.embed_dim))
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h2 = self.transformer2(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        h5 = self.conv5(h4)
        h5 += self.dense5(embed)
        h5 = self.gnorm5(h5)
        h5 = self.act(h5)

        h6 = self.transformer5(h5)

        h = self.tconv5(h5)
        h += self.dense8(embed)
        h = self.tgnorm5(h)
        h = self.act(h)
        h = self.tconv4(torch.cat([h, h4], dim=1))
        h += self.dense9(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense10(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.transformer10(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense11(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h
        return h
