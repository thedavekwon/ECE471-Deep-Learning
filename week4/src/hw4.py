import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

from torch.utils import data
from torchsummary import summary
from ResNet import *
from radam import RAdam

"""
Initially, I have tried with a model similar to convnet, where is had four convolutional
layers with maxpooling and batch normalization and three dense layers. I was able to reach
84.32% accuracy in the CIFAR10 and 83.15% top 5 accuray in CIFAR100. And then, I
want to learn and try to implement ResNet, so I looked at the tutorial to build ResNet hoping
that the deeper model (resnet101 and resnet151) will able to achieve higher accuracy.
With resnet101 I was able to reach 89.4% accuracy in the CIFAR10. I wasn't able to recognize
the resnet101 and resnet151 are overfitting, because I was only getting an accuracy of the test.
So I tried to adding Dropout to resnet building blocks and increasing lambda for the
l2-regularization. This helped the overfitting, but due to the limitation of computation, the
training took so long. Also, I tried to look into data augmentation at the last moment, but 
I wasn't able to find easy to use data augmentation library in PyTorch using the PyTorch dataset
class. But I believe with data augmentation, I would have able to leverage accuracy a 
little bit further. Also, I tried to use transfer learning by utilizing pre-trained weight
and unfreezing last fully connected layer or few convolutional layers at the back to 
fine-tune the transfer learning.  I thought that the CNN of pretrained resnet, shufflenet,
or those variants will provide generalized feature extraction, but I wasn't able to achieve a
good results with it. 

I have made several mistakes while I was exploring the models. First of all, I was using 
batch size that was too big. I started with a batch size of 2500, which is 1/18 of the 
training set. I naively thought that the bigger batch size will be beneficial to the 
computing speed without considering bigger batch size tends to fail in generalization. 
Secondly, I naively thought Adam was the best optimizer that would always optimize better 
than SGD or RMSProp, but through research, I found that Adam is more robust but SGD might 
have a better results if it has correct hyper-parameter settings. Also, I found a new
optimizer that was released last month called RAdam, which has a dynamic adjustment
to the adaptive learning rate based on variance and momentum during training.
I found the PyTorch library that the authors of the paper released 
and tried to use it with resnet and my original model.(https://arxiv.org/abs/1908.03265)
Also, I should have used a PyTorch library that saves weight so that I could easily
re-load the weights when I want to test out or train more iterations. 

The accuracy plot of resnet101 has a training label, but I made a mistake of labeling as
training. I originally had the training and test accuracy, but I removed training 
accuracy because I remember that the softmax was an expensive operation. I didn't have
enough time to re-run the experiment. From now on, I learned that I should save the
parameters weight whenever I train other models. 
"""


def load_CIFAR10_data():
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, download=True, transform=train_transform
    )
    train_set, validation_set = data.random_split(
        train_set, (int(len(train_set) * 0.9), int(len(train_set) * 0.1))
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=test_transform
    )

    train_loader = data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )
    validation_loader = data.DataLoader(
        validation_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_loader, validation_loader, test_loader, test_set.classes


def load_CIFAR100_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = torchvision.datasets.CIFAR100(
        root="./dataset", train=True, download=True, transform=transform
    )
    train_set, validation_set = data.random_split(
        train_set, (int(len(train_set) * 0.9), int(len(train_set) * 0.1))
    )
    test_set = torchvision.datasets.CIFAR100(
        root="./dataset", train=False, download=True, transform=transform
    )

    train_loader = data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )
    validation_loader = data.DataLoader(
        validation_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )

    return train_loader, validation_loader, test_loader, test_set.classes


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def accuracy_plot(accuracies, TEST):
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel("epoch")
    plt.ylabel("accuracy\n(%)").set_rotation(0)

    if TEST:
        plt.legend(["train", "test"])
    else:
        plt.legend(["train", "validation"])


def losses_plot(train_losses, losses, TEST):
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss").set_rotation(0)

    if TEST:
        plt.legend(["train", "test"])
    else:
        plt.legend(["train", "validation"])


def train(model, train_loader, optimizer, criterion, device, epoch, topk):
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
        if topk == 1:
            pred = outputs.argmax(dim=1, keepdim=True)
            correct_pred = pred.eq(y.view_as(pred))
            correct += correct_pred.sum().item()
        else:
            _, pred = outputs.topk(topk, 1, True, True)
            correct_pred = pred.eq(y.view(-1, 1).expand_as(pred))
            correct += correct_pred.sum().item()
    train_percentage = round(correct / len(train_loader.dataset) * 100, 2)
    print(
        f"Epoch:{epoch} Train loss: {loss.item():0.6f}, Train Accuracy:{correct}/{len(train_loader.dataset)} ({train_percentage}%)"
    )
    return loss.item()


def validate(model, test_loader, criterion, device, epoch, classes, topk):
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
            if (topk == 1):
                pred = outputs.argmax(dim=1, keepdim=True)
                correct_pred = pred.eq(y.view_as(pred))
                correct += correct_pred.sum().item()
            else:
                _, pred = outputs.topk(topk, 1, True, True)
                correct_pred = pred.eq(y.view(-1, 1).expand_as(pred))
                correct += correct_pred.sum().item()
            
    test_percentage = round(correct / len(test_loader.dataset) * 100, 2)
    print(
        f"Epoch:{epoch} Validation loss: {test_loss:0.6f}, Validation Accuracy:{correct}/{len(test_loader.dataset)} ({test_percentage}%)"
    )
    return test_percentage, test_loss


def test(model, test_loader, criterion, device, epoch, classes, topk):
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
            if topk == 1:
                pred = outputs.argmax(dim=1, keepdim=True)
                correct_pred = pred.eq(y.view_as(pred))
                correct += correct_pred.sum().item()
            else:
                _, pred = outputs.topk(topk, 1, True, True)
                correct_pred = pred.eq(y.view(-1, 1).expand_as(pred))
                correct += correct_pred.sum().item()
    test_loss /= len(test_loss.dataset)
    test_percentage = round(correct / len(test_loader.dataset) * 100, 2)
    print(
        f"Epoch:{epoch} Test loss: {test_loss:0.6f}, Test Accuracy:{correct}/{len(test_loader.dataset)} ({test_percentage}%)"
    )
    return test_percentage, test_loss


class convLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        pooling_size,
        padding_size,
        dropout_rate,
    ):
        super(convLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=padding_size
        )
        self.pooling_size = pooling_size
        if self.pooling_size:
            self.pool = nn.MaxPool2d(pooling_size)
        self.bm = nn.BatchNorm2d(out_channel)
        self.drop = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bm(x)
        if self.pooling_size:
            x = self.pool(x)
        x = self.drop(x)
        return x


# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CIFAR10ResModel(nn.Module):
    def __init__(self):
        super(CIFAR10ResModel, self).__init__()
        self.res = resnet(3, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.layers = [self.res, self.fc1, self.fc2, self.fc3]
        self.activations = [True, True, True, True]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            if activation:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class CIFAR10ResLayer(nn.Module):
    def __init__(self):
        super(CIFAR10ResLayer, self).__init__()
        self.conv1 = convLayer(3, 64, 3, 2, 1, 0.3)
        self.conv2 = convLayer(64, 128, 3, 2, 1, 0.3)
        self.conv3 = convLayer(128, 256, 5, 2, 1, 0.3)
        self.res1 = ResNetLayer(256, 512, n=3)
        self.conv4 = convLayer(512, 512, 2, 2, 1, 0.4)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.res1,
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
        ]
        self.activations = [False, False, False, False, False, True, True, True]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            if activation:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = convLayer(3, 64, 3, 2, 1, 0.3)
        self.conv2 = convLayer(64, 128, 3, 0, 1, 0.3)
        self.conv3 = convLayer(128, 256, 5, 2, 1, 0.3)
        self.conv4 = convLayer(256, 512, 5, 2, 1, 0.3)
        self.conv5 = convLayer(512, 512, 3, 0, 1, 0.4)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
        ]
        self.activations = [False, False, False, False, False, False, True, True, True]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            if activation:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class CIFAR10ModelTest(nn.Module):
    def __init__(self):
        super(CIFAR10ModelTest, self).__init__()
        self.conv1 = convLayer(3, 64, 3, 2, 1, 0.3)
        self.conv2 = convLayer(64, 128, 3, 0, 1, 0.3)
        self.conv3 = convLayer(128, 256, 5, 2, 1, 0.3)
        self.conv4 = convLayer(256, 512, 5, 0, 1, 0.3)
        self.conv5 = convLayer(512, 512, 1, 2, 1, 0.4)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(4608, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)
        self.drop = nn.Dropout(0.25)
        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4,
        ]
        self.activations = [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            if activation:
                x = self.drop(F.relu(layer(x)))
            else:
                x = layer(x)
        return x


class CIFAR100Model(nn.Module):
    def __init__(self):
        super(CIFAR100Model, self).__init__()
        self.conv1 = convLayer(3, 64, 3, 2, 1, 0.4)
        self.conv2 = convLayer(64, 128, 3, 0, 1, 0.4)
        self.conv3 = convLayer(128, 256, 5, 2, 2, 0.4)
        self.conv4 = convLayer(256, 512, 5, 2, 2, 0.4)
        self.conv5 = convLayer(512, 512, 1, 0, 0, 0.4)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 100)
        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
        ]
        self.activations = [False, False, False, False, False, False, True, True, True]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            if activation:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


def transfer_radam_densenet(num_classes):
    model = torchvision.models.densenet161(pretrained=True)
    #     for param in model.parameters():
    #         param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.fc.parameters())
    return model, criterion, optimizer


def transfer_radam_shufflenet(num_classes):
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.fc.parameters())
    return model, criterion, optimizer


def transfer_radam_resnet101(num_classes):
    model = torchvision.models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.fc.parameters())
    return model, criterion, optimizer


def transfer_learning_resnet152(num_classes):
    model = torchvision.models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.fc.parameters())
    return model, criterion, optimizer


EPOCH = 200
TEST = True
LAMBDA = 0.02

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader, classes = load_CIFAR10_data()

    # model = CIFAR10Model()
    # model = CIFAR10ModelTest()
    # model = resnet18(3, 10)
    # model = CIFAR10ResLayer()
    # model = CIFAR100Model()
    # model, criterion, optimizer = transfer_radam_resnet101(100)

    model = resnet18(3, 10, drop=False)
    summary(model, (3, 32, 32), device="cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters())
    # optimizer = optim.Adam(model.parameters())

    accuracies = []
    train_losses = []
    losses = []

    for epoch in range(1, EPOCH + 1):
        loss = train(model, train_loader, optimizer, criterion, device, epoch, 1)
        train_losses.append(loss)
        if not TEST:
            acc, loss = validate(
                model, validation_loader, criterion, device, epoch, classes, 1
            )
            accuracies.append(acc)
            losses.append(loss)
        else:
            acc, loss = test(model, test_loader, criterion, device, epoch, classes, 1)
            accuracies.append(acc)
            losses.append(loss)
    plt.figure(1)
    accuracy_plot(accuracies, TEST)
    plt.figure(2)
    losses_plot(train_losses, losses, TEST)
    plt.show()
