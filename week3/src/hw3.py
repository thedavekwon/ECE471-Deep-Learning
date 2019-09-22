import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

EPOCH = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
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

    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=4)

    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                              shuffle=False, num_workers=4)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = Model()
    summary(model, (1, 28, 28))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, EPOCH + 1):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
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
                outputs = model(data)
                print(outputs, target)
                test_loss += criterion(outputs, target).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        percentage = correct / len(test_loader.dataset)
        print(f"Average loss: {test_loss}, Accuracy:{correct}/{len(test_loader.dataset)} ({percentage}%)")
