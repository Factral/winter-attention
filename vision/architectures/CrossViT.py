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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self,x):
        x = self.MLP(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        #head_dims = dim // n_heads

        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

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
    def __init__(self, dim, n_heads, mlp_ratio=4., has_mlp=False):
        super().__init__()
        self.has_mlp = has_mlp

        self.ln1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, n_heads = n_heads)
        self.ln2 = nn.LayerNorm(dim)
        if has_mlp:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MultiLayerPerceptron(dim, mlp_hidden_dim)

    def forward(self, x):
        x = x[:,0,...] + self.attn(self.ln1(x))

        if self.has_mlp:
            x = x + self.mlp(self.ln2(x))
        return x
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_hidden_dim):
        super().__init__()

        self.att = CrossAttention(dim, n_heads)
        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.mlp = MultiLayerPerceptron(dim, mlp_hidden_dim)

    def forward(self, x):

        x1 = self.ln1(x) #layer normalization
        x1 = self.att(x1) # multihead attention

        x = x + x1 # residual connection

        x2 = self.ln2(x) #layer normalization
        x2 = self.mlp(x2) # multilayer perceptron

        y = x + x2 # residual connection

        return y

class MultiScaleTransformerEncoder(nn.Module):
    def __init__(self, dim, patches, n_heads, depths, mlp_ratio):
        super().__init__()

        num_branches = len(dim)
        mlp_hidden_dims = dim * mlp_ratio
        self.num_branches = num_branches

        self.blocks = nn.ModuleList([])
        for branch in range(num_branches):
            self.blocks = nn.Sequentia(*[BasicTransformerBlock(dim=dim[branch], n_heads=n_heads[branch], mlp_hidden_dim=mlp_hidden_dims) for _ in range(depths)])

        self.projs = nn.ModuleList([])
        for branch in range(num_branches):
            self.projs = nn.Sequential(*[nn.LayerNorm(dim[branch]), nn.GELU(), nn.Linear(dim[branch], dim[(branch+1) % num_branches])])

        self.fusion = nn.ModuleList([])
        for branch in range(num_branches):
            self.fusion = nn.Sequential(*[])
