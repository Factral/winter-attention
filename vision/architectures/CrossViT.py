import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from einops import rearrange, repeat
from ..utils import to_2tuple


class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #B, C, H, W = x.size()
        x = self.proj(x)
        x = rearrange(x, 'b e n1 n2 -> b (n1 n2) e') # b, long, embedding

        return x
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
        )

    def forward(self,x):
        x = self.MLP(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()

        self.n_heads = n_heads
        #head_dims = n_embed // n_heads

        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)

        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        B, L, E = x.size()
        q = rearrange(self.query(x[:,0,...]), 'b 1 e -> b h 1 f', h = self.n_heads , f = E/self.n_heads) 
        k = rearrange(self.key(x), 'b l e  -> b h l f', h = self.n_heads, f = E/self.n_heads) 
        v = rearrange(self.value(x), 'b l e -> b h l f ', h = self.n_heads, f = E/self.n_heads)

        scores = q @ k.transpose(-2,-1) / torch.sqrt(k.size(-1)) 
        att = F.softmax(scores, dim=-1)

        y = att @ v
        y = rearrange(y, 'b h 1 f -> b 1 (h f)')
        y = self.proj(y)

        return y 
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, n_embed, n_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads

        mlp_hidden_dim = int(n_embed * mlp_ratio)