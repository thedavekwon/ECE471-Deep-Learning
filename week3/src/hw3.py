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

def save(path, transform):
    images = [transform(io.imread(os.path.join(path, file))) for file in os.listdir(path)]
    labels = [int(file[-5]) for file in os.listdir(path)]
    with open(path+".p", "wb") as f:
        pickle.dump({"images":images, "labels":labels}, f)

def load(path):
    with open(path+".p", "rb") as f:
        tmp = pickle.load(f)
    return tmp["images"], tmp["labels"]

# transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.1307,), (0.3081,))]
#             )
# save("dataset/MNIST/train", transform)
# save("dataset/MNIST/test", transform)

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
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4096,
                                               shuffle=True, num_workers=3)
    vali_loader = torch.utils.data.DataLoader(vali_set, batch_size=4096,
                                              shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4096,
                                              shuffle=False, num_workers=3)
    return train_loader, vali_loader, test_loader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
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

class EfficientModel(nn.Module):
    def __init__(self):
        super(EfficientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 1)
        self.bm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 1, 3, stride=2)
        # self.bm2 = nn.BatchNorm2d(1)
        # self.conv3 = nn.Conv2d(4, 1, 1)
        
        self.fc1 = nn.Linear(9, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(9, 10)

        self.pool = nn.MaxPool2d(2)
        self.avgPool = nn.AvgPool2d(2)
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.pool(self.drop(self.bm1(F.relu(self.conv1(x)))))
        # x = self.pool(self.drop(self.bm2(F.relu(self.conv2(x)))))
        x = self.avgPool(self.drop(F.relu(self.conv2(x))))

        # x = self.avgPool(self.drop(F.relu(self.conv3(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        return functools.reduce(lambda a, b: a * b, x.size()[1:])

LEARNING_RATE = 0.001
EPOCH = 100

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, vali_loader, test_loader = load_data()
    
    # model = Model()
    model = EfficientModel()
    model.to(device)
    
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(1, EPOCH + 1):
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        vali_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in vali_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                vali_loss += criterion(outputs, target).item()
                outputs = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        vali_loss /= len(vali_loader.dataset)
        percentage = round(correct / len(vali_loader.dataset)*100, 2)
        print(f"Average Train loss: {train_loss:0.6f}, Average Validation loss: {vali_loss:0.6f}, Accuracy:{correct}/{len(vali_loader.dataset)} ({percentage}%)")