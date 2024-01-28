import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import datasets, transforms
from einops import rearrange, repeat, einsum


class PatchEmbedding(nn.Module):
    """
        This is equivalent to patchMerging in SwinTransformer
        see: https://github.com/microsoft/Swin-Transformer/issues/256
    """
        
    def __init__(self, patch_size=4, in_chans=3, c=96):

        super().__init__()
        self.proj = nn.Conv2d(in_chans, c, kernel_size=patch_size, stride=patch_size)
        # b, channels, n_patches_h, n_patches_w

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c n1 n2 -> b n1 n2 c') # b, n_w, n_h, c
        return x
    

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))
    

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')


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
    

class W_MSA(nn.Module):
    def __init__(self, dim, heads, shifted, window_size):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.shifted = shifted
        self.window_size = window_size

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)


        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.pos_emb = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

    def forward(self, x):
        b, n_h, n_w, c = x.shape
        assert c % h == 0, 'dimensions must be divisible by number of heads'

        if self.shifted:
            x = self.cyclic_shift(x)

        h = self.heads
        
        # batch, n_h, n_w, dim

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        # batch, n_h, n_w, head_dim * 3 

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w) (nw_w w) (h d) -> b h (nw_h nw_w) (w w) d',
                                 h=h, w=self.windows_size), qkv)

        # batch, heads, n_patches, windows_size^2, head_dim
        
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * (k.size(-1) ** -0.5)

        #batch, heads, n_patches, windows_size^2, windows_size^2

        dots += self.pos_emb

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i d, b h w d j -> b h w i j', attn, v)
        
        # batch, heads, n_patches, windows_size^2, head_dim

        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)

        # batch, n_h, n_w, dim

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, shifted, window_size, mlp_dim):
        super().__init__()
        self.msa = W_MSA(dim, heads, shifted, window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = self.msa(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x


class Stage(nn.Module):
    def __init__(self, in_chans, hidden_dim, layers, downscaling_factor, heads, window_size):
        super().__init__()
        assert layers % 2 == 0, 'stage layers must be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchEmbedding(patch_size=downscaling_factor, in_chans=in_chans, c=hidden_dim)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(SwinBlock(dim=hidden_dim, heads=heads, shifted=False, window_size=window_size,
                                         mlp_dim=hidden_dim * 4))
            self.layers.append(SwinBlock(dim=hidden_dim, heads=heads, shifted=True, window_size=window_size,
                                         mlp_dim=hidden_dim * 4))
            
    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x
    

class SwinTransformer(nn.Module):
    def __init__(self, in_chans, hidden_dim, layers, heads, downscaling_factors, window_size, num_classes):
        super().__init__()

        self.stages = nn.ModuleList([])
        for i in range(len(layers)):
            self.stages.append(
                Stage(  in_chans = in_chans if i == 0 else hidden_dim * 2,
                        hidden_dim = hidden_dim  if i == 0 else hidden_dim * 2,
                        layers = layers[i],
                        downscaling_factor = downscaling_factors[i],
                        heads = heads[i],
                        window_size = window_size))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)

        x = x.mean(dim=[2,3])

        return self.mlp_head(x)
    