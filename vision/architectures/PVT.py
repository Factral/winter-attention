import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from einops import rearrange, repeat
import math



class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # b, embedding, n_patches_h, n_patches_w
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, x):
        B, C, H, W = x.size()

        x = self.proj(x)
        x = rearrange(x, 'b e n1 n2 -> b (n1 n2) e') # b, long, embedding
        x = self.norm(x)

        n_patches_h, n_patches_w = H // self.patch_size, W // self.patch_size

        return x, (n_patches_h, n_patches_w)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            #nn.Dropout(drop)
        )

    def forward(self,x):
        x = self.MLP(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_heads, reduction_ratio):
        super().__init__()

        # key, query, value projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.n_heads = n_heads

        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio  > 1:
            self.sr = nn.Conv2d(n_embd, n_embd, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(n_embd)

    def forward(self, x, H, W):
        # B, L, F = x.size() # batch, length, features

        q = rearrange(self.query(x), 'b l (h f) -> b h l f', h=self.n_heads) # (B, L, F) -> (B, H, L, F/H)
        
        if self.reduction_ratio > 1:
            # f = h*w

            x = rearrange(x, 'b (h w) f -> b f h w', h=H, w=W)
            x = self.sr(x)
            x = rearrange(x, 'b e n1 n2 -> b (n1 n2) e')
            x = self.norm(x)

        k = rearrange(self.key(x), 'b l (h f) -> b h l f', h=self.n_heads) # (B, L, F) -> (B, H, L, F/H)
        v = rearrange(self.value(x), 'b l (h f) -> b h l f', h=self.n_heads) # (B, L, F) -> (B, H, L, F/H)

        scores = q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)) # B, H, L, L
        att = F.softmax(scores, dim=-1)

        y = att @ v # B, H, L, F/H

        y = rearrange(y, 'b h l f -> b l (h f)')


        y = self.proj(y) # batch, length, feature

        return y


class TransformerEncoder(nn.Module):
    def __init__(self, n_embd, n_heads, reduction_ratio, expansion_ratio):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.att = MultiHeadAttention(n_embd, n_heads, reduction_ratio)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        hidden_dims_mlp = expansion_ratio * n_embd
        self.mlp = MultiLayerPerceptron(n_embd, hidden_dim=hidden_dims_mlp)

    def forward(self, x, H, W):

        x1 = self.ln1(x) #layer normalization
        x1 = self.att(x1, H, W) # multihead attention

        x = x + x1 # residual connection

        x2 = self.ln2(x) #layer normalization
        x2 = self.mlp(x2) # multilayer perceptron

        y = x + x2 # residual connection

        return y


class PVT(nn.Module):

    def __init__(self,img_dim , in_chans, patch_dim, num_stages, embed_dims,
                 encoder_layers, reduction_ratio, n_heads,
                    expansion_ratio, num_classes):
        super().__init__()

        self.num_stages = num_stages
        self.num_classes = num_classes
        self.encoder_layer = encoder_layers #model depth
        
        #self.patch_embed = PatchEmbedding(embed_dims[0], 1, embed_dims[0])


        for i in range(num_stages):
            patch_embed = PatchEmbedding(patch_size=patch_dim if i == 0 else 2,
                                         in_chans=in_chans if i == 0 else embed_dims[i-1],
                                         embed_dim=embed_dims[i])
            
            img_dim = 256 // (2  ** (i+1)) if i != 0 else img_dim
            n_patches = (img_dim // patch_dim) ** 2 if i == 0 else (img_dim // 2) ** 2
            n_patches = n_patches if i != num_stages - 1 else n_patches + 1

            pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dims[i]) ) 

            transformer_block = nn.ModuleList([
                TransformerEncoder(n_embd=embed_dims[i],n_heads=n_heads[i],
                                    reduction_ratio=reduction_ratio[i], expansion_ratio=expansion_ratio[i])
                                    for _ in range(encoder_layers[i])])
            
            setattr(self, f'patch_embed_{i}', patch_embed)
            setattr(self, f'pos_embed_{i}', pos_embed)
            setattr(self, f'transformer_block_{i}', transformer_block)

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
        self.MLP_head = nn.Linear(embed_dims[-1], num_classes)



    def forward(self, x):

        B = x.size(0)

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed_{i}')
            pos_embed = getattr(self, f'pos_embed_{i}')
            transformer_block = getattr(self, f'transformer_block_{i}')

            x, (H, W) = patch_embed(x)
            
            #print(H,W)

            if i == self.num_stages - 1:
                cls_tokens = repeat(self.cls_token, '() 1 e -> b 1 e', b=B)
                x = torch.cat((cls_tokens, x), dim=1) # (B, L+1, F)
            
            print(x.shape, pos_embed.shape)
            x = x + pos_embed
            for blk in transformer_block:
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                x = rearrange(x, 'b (h w) f -> b f h w', h=H, w=W) 


        x = x[:, 0, :]
        #x = self.norm(x)

        x = self.MLP_head(x)


        return x


