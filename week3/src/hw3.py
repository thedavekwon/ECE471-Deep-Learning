import functools
import os
import pickle

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

import matplotlib.pyplot as plt


def save(path, transform):
    images = [
        transform(io.imread(os.path.join(path, file))) for file in os.listdir(path)
    ]
    labels = [int(file[-5]) for file in os.listdir(path)]
    with open(path + ".p", "wb") as f:
        pickle.dump({"images": images, "labels": labels}, f)


def load(path):
    with open(path + ".p", "rb") as f:
        tmp = pickle.load(f)
    return tmp["images"], tmp["labels"]


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# if "train.p" not in os.listdir("dataset/MNIST"):
#     save("dataset/MNIST/train", transform)
# if "test.p" not in os.listdir("dataset/MNIST"):
#     save("dataset/MNIST/test", transform)


class MNISTDataset(data.Dataset):
    def __init__(self, path):
        super(MNISTDataset).__init__()
        self.images, self.labels = load(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_data():
    train_set = MNISTDataset(path="dataset/MNIST/train")
    train_set, vali_set = data.random_split(train_set, (54000, 6000))
    test_set = MNISTDataset(path="dataset/MNIST/test")

    print(len(train_set), len(vali_set), len(test_set))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4096, shuffle=True, num_workers=3
    )
    vali_loader = torch.utils.data.DataLoader(
        vali_set, batch_size=4096, shuffle=True, num_workers=3
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=4096, shuffle=False, num_workers=3
    )
    return train_loader, vali_loader, test_loader

# accuracy of 98.5%
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1, padding=1)
        self.bm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.bm3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.4)

        self.fc1 = nn.Linear(144, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.drop(self.bm1(F.relu(self.conv1(x)))))
        x = self.pool(self.drop(self.bm2(F.relu(self.conv2(x)))))
        x = self.pool(self.drop(self.bm3(F.relu(self.conv3(x)))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return functools.reduce(lambda a, b: a * b, x.size()[1:])

#312 parameters
class EfficientModel(nn.Module):
    def __init__(self):
        super(EfficientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 1)
        self.bm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 1, 3)
        self.conv3 = nn.Conv2d(1, 4, 3)
        self.bm3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 1, 1)

        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 10)
        # self.fc3 = nn.Linear(9, 10)

        self.pool = nn.MaxPool2d(2)
        self.avgPool = nn.AvgPool2d(2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(self.drop(self.bm1(F.elu(self.conv1(x)))))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.pool(self.bm3(F.relu(self.conv3(x))))
        x = self.pool(F.elu(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        return functools.reduce(lambda a, b: a * b, x.size()[1:])


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    correct = 0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        regularization_loss = 0.0
        for param in model.parameters():
            regularization_loss += torch.norm(param)
        loss = criterion(outputs, y) + LAMBDA * regularization_loss
        loss.backward()
        optimizer.step()
        outputs = F.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    train_percentage = round(correct / len(train_loader.dataset) * 100, 2)
    print(f"Train Accuracy:{correct}/{len(train_loader.dataset)} ({train_percentage}%)")
    return train_percentage, loss.item()

def validate(model, vali_loader, optimizer, criterion, device):
    model.eval()
    vali_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in vali_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            vali_loss += criterion(outputs, y).item()
            outputs = F.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    vali_loss /= len(vali_loader.dataset)
    vali_percentage = round(correct / len(vali_loader.dataset) * 100, 2)
    print(
        f"Validation loss: {vali_loss:0.6f}, Validation Accuracy:{correct}/{len(vali_loader.dataset)} ({vali_percentage}%)"
    )
    return vali_percentage, vali_loss


def test(model, test_loader, optimizer, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            test_loss += criterion(outputs, y).item()
            outputs = F.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_percentage = round(correct / len(test_loader.dataset) * 100, 2)
    print(
        f"Test loss: {test_loss:0.6f}, Test Accuracy:{correct}/{len(test_loader.dataset)} ({test_percentage}%)"
    )
    return test_percentage, test_loss

def accuracy_plot(train_accuracies, accuracies, TEST):
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.xlabel("epoch")
    plt.ylabel("accuracy\n(%)").set_rotation(0)
    
    
    if (TEST):
        plt.legend(["train", "test"])
    else:
        plt.legend(["train", "validation"])
    plt.show()
    
def losses_plot(train_losses, losses, TEST):
    plt.plot(range(1, len(train_losses)+1), train_losses)
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss").set_rotation(0)
    
    
    if (TEST):
        plt.legend(["train", "test"])
    else:
        plt.legend(["train", "validation"])

LEARNING_RATE = 0.001
EPOCH = 50
LAMBDA = 0.003
TEST = True

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, vali_loader, test_loader = load_data()

    model = Model()
    # model = EfficientModel()
    model.to(device)

    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_accuracies = []
    accuracies = []
    train_losses = []
    losses = []
    for epoch in range(1, EPOCH + 1):
        acc, loss = train(model, train_loader, optimizer, criterion, device)
        train_accuracies.append(acc)
        train_losses.append(loss)
        if not TEST:
            acc, loss = validate(model, vali_loader, optimizer, criterion, device)
            accuracies.append(acc)
            losses.append(loss)
        else:
            acc, loss = test(model, vali_loader, optimizer, criterion, device)
            accuracies.append(acc)
            losses.append(loss)
    plt.figure(1)
    accuracy_plot(train_accuracies, accuracies, TEST)
    plt.figure(2)
    losses_plot(train_losses, losses, TEST)
    plt.show()