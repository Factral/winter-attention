import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from einops import rearrange, repeat
from model_loader import load_model

import math
import argparse
import tqdm as tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Selection Script')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--resize', type=int, default=0, help='Resize the images to a square of this size')
    parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help='Eval after n epochs')
    args = parser.parse_args()

    # Cargar el modelo usando la función en otro archivo
    model = load_model(args.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.resize != 0:
        preprocess = transforms.Compose([
            transforms.Resize((args.resize ,args.resize )),
            transforms.ToTensor()
            ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor()
            ])

    train_dataset = datasets.CIFAR10(root='./data/',train=True, transform=preprocess, download=True)
    test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=preprocess, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Training model: %s" % args.model_name)
    print("Model parameters: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))


    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):

        model.train()
        train_loss = 0
        train_iter = 0
        for _, batch in enumerate(tqdm(train_loader, desc = "Epoch")):
            batch = tuple(b.to(device) for b in batch)
            images, labels = batch

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_iter += 1

        train_loss = loss / train_iter
        correct = 0
        total = 0

        if epoch % args.eval_interval == 0:
            total, correct = 0, 0
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(tqdm(test_loader, desc = "Epoch")):
                    batch = tuple(b.to(device) for b in batch)
                    images, labels = batch
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

            acc = (correct / total).item()
            print('\nEval at epoch: %d train_loss: %f Accuracy: %f' % (epoch, train_loss, acc))
   







