import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from einops import rearrange, repeat
from model_loader import load_model

import math
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Selection Script')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--resize', type=int, default=0, help='Resize the images to a square of this size')
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


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(5):

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 100 == 0:
                print("Epoch: %d, Batch: %d, Loss: %f" % (epoch, i, loss.item()))


        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        print("Epoch: %d, Accuracy: %f" % (epoch, 100 * correct / total))

