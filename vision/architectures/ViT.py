import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from einops import rearrange, repeat
import math


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=7, in_chans=1, embed_dim=49):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # b, embedding, n_patches_h, n_patches_w

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b e n1 n2 -> b (n1 n2) e') # b, long, embedding
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()

        # key, query, value projections
        self.key = nn.Linear(n_embd, n_embd * n_heads)
        self.query = nn.Linear(n_embd, n_embd * n_heads)
        self.value = nn.Linear(n_embd, n_embd * n_heads)
        self.n_heads = n_heads

        # output projection
        self.proj = nn.Linear(n_embd * n_heads, n_embd)


    def forward(self, x):
        # B, L, F = x.size() # batch, length, features

        k = rearrange(self.key(x), 'b l (h f) -> b h l f', h=self.n_heads) # (B, L, F) -> (B, H, L, F/H)
        q = rearrange(self.query(x), 'b l (h f) -> b h l f', h=self.n_heads) # (B, L, F) -> (B, H, L, F/H)
        v = rearrange(self.value(x), 'b l (h f) -> b h l f', h=self.n_heads) # (B, L, F) -> (B, H, L, F/H)

        scores = q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)) # B, H, L, L
        att = F.softmax(scores, dim=-1)

        y = att @ v # B, H, L, F/H

        y = rearrange(y, 'b h l f -> b l (h f)')


        y = self.proj(y) # batch, length, feature

        return y


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd)
        )

    def forward(self,x):
        x = self.MLP(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()

        self.att = MultiHeadAttention(n_embd, n_heads)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        self.mlp = MultiLayerPerceptron(n_embd, n_embd)

    def forward(self, x):

        x1 = self.ln1(x) #layer normalization
        x1 = self.att(x1) # multihead attention

        x = x + x1 # residual connection

        x2 = self.ln2(x) #layer normalization
        x2 = self.mlp(x2) # multilayer perceptron

        y = x + x2 # residual connection

        return y


class ViT(nn.Module):

    def __init__(self,img_dim , patch__dim=7, embed_dim=100, num_classes=10, n_heads=8, depth=2):
        super().__init__()


        self.patch_embed = PatchEmbedding(patch__dim, 1, embed_dim)
        self.cls_token = nn.Parameter( torch.randn(1, 1, embed_dim))

        n_patches = (img_dim // patch__dim) ** 2
        self.pos_embedding = nn.Parameter( torch.randn(1, n_patches+1, embed_dim))

        self.transformer = nn.Sequential(*[TransformerEncoder(embed_dim, n_heads) for _ in range(depth)])
        self.MLP_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        e = self.patch_embed(x) # (B, L, F)

        #cls token addition
        cls = repeat(self.cls_token, '1 1 f -> b 1 f', b=e.shape[0])
        e = torch.cat([cls, e], dim=1) + self.pos_embedding # (B, L+1, F)

        e += self.pos_embedding

        z = self.transformer(e) # (B, L+1, F)

        cls_token = z[:, 0, :]  # (B, F)

        y = self.MLP_head(cls_token) # (B, num_classes)

        return y



model = ViT(img_dim=28,
            patch__dim=7, 
            embed_dim=49, 
            num_classes=10, 
            n_heads=3, 
            depth=2)
