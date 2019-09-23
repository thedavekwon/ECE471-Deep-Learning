import functools
import os

from skimage import io
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils import data

EPOCH = 20


class MNISTDataset(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.images = [transform(io.imread(os.path.join(self.path, file))) for file in os.listdir(path)]
        self.labels = [int(file[-5]) for file in os.listdir(path)]
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(144, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return functools.reduce(lambda a, b: a * b, x.size()[1:])


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_set = MNISTDataset(path="dataset/MNIST/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2048,
                                               shuffle=True, num_workers=2)

    test_set = MNISTDataset(path="dataset/MNIST/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048,
                                              shuffle=False, num_workers=2)
    return train_loader, test_loader


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_loader, test_loader = load_data()
    model = Model()
    model.to(device)
    summary(model, (1, 28, 28))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, EPOCH + 1):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch}, {i + 1}] loss:{running_loss / 2000}")
                running_loss = 0.0

        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                test_loss += criterion(outputs, target).item()
                outputs = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        percentage = correct / len(test_loader.dataset)
        print(f"Average loss: {test_loss:0.6f}, Accuracy:{correct}/{len(test_loader.dataset)} ({percentage}%)")