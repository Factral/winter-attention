import torch
import torch.nn as nn

from torchvision import datasets, transforms
from einops import rearrange
import math

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

class mnistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        patches = rearrange(img, '1 (h p1) (w p2) -> (h w) (p1 p2 1)', p1=7, p2=7)
        return patches, label

    def __len__(self):
        return len(self.dataset)
    
train_loader = torch.utils.data.DataLoader(dataset=mnistDataset(train_dataset), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnistDataset(test_dataset), batch_size=64, shuffle=True)


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # key, query, value projections
        self.key = torch.nn.Linear(n_embd, n_embd*n_heads)
        self.query = torch.nn.Linear(n_embd, n_embd*n_heads)
        self.value = torch.nn.Linear(n_embd, n_embd*n_heads)

        # output projection
        self.proj = torch.nn.Linear(n_embd*n_heads, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = rearrange(self.key(x), 'b l (nh f) -> b nh l f', nh=self.n_heads)
        q = rearrange(self.query(x), 'b l (nh f) -> b nh l f', nh=self.n_heads)
        v = rearrange(self.value(x), 'b l (nh f) -> b nh l f', nh=self.n_heads)

        # attention (B, nh, L, F) x (B, nh, F, L) -> (B, nh, L, L)
        att = torch.einsum('bnlf, bnaf -> bnla', q, k) / math.sqrt(k.size(-1))

        att = torch.nn.functional.softmax(att, dim=-1)

        y = att @ v # (B, nh, L, L) x (B, nh, L, F) -> (B, nh, L, F)

        y = rearrange(y, 'b nh l f -> b l (nh f)')

        return self.proj(y)

class Model(nn.Module):

    def __init__(self, n_embd=7*7, seq_len=4*4, n_heads=4*4):
        super().__init__()
        self.mlp = None

        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.actn = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(n_embd*seq_len, 10)

    def forward(self, x):
        x = self.attn(x)
        x = rearrange(x, 'b l f -> b (l f)')
        x = self.actn(x)
        y = self.fc(x)
        #print(y.shape)
        return y

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(5):

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch: %d, Batch: %d, Loss: %f" % (epoch, i, loss.item()))


    correct = 0
    total = 0
    
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print("Epoch: %d, Accuracy: %f" % (epoch, 100 * correct / total))
