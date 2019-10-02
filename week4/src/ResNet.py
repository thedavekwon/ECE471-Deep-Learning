# Tutorial for ResNet from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
import numpy as np
import torch
import torch.nn as nn

from torchsummary import summary
from functools import partial

# dyanamic padding for preserving size
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels))


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        if self.residual:
            identity = self.shortcut(x)
        x = self.blocks(x)
        x += identity
        x = self.activate(x)
        return x
        
    @property
    def residual(self):
        return self.in_channels != self.out_channels

class ResNetBlock(Block):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        if self.residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                          stride=self.downsampling, bias=False),
                nn.BatchNorm2d(self.expanded_channels)
            ) 
        else:
            self.shortcut = None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def residual(self):
        return self.in_channels != self.expanded_channels
    
class ResNetBasicBlock(ResNetBlock):
    expansion = 1
    """
    two layers of 3x3 conv/batch norm/activation
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False)
        )


class ResNetBottleNeckBlock(ResNetBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBottleNeckBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels*block.expansion, out_channels, downsampling=1,
                    *args, **kwargs) for _ in range(n-1)]
        )
    
    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoderNoMaxPool(nn.Module):
    def __init__(self, in_channels=3, block_sizes=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2], activation='relu', block=ResNetBasicBlock,
                 *args, **kwargs):
        super().__init__()
        self.block_sizes = block_sizes
        self.activation=activation
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation_func(activation),
        )
        
        self.in_out_block_sizes = list(zip(block_sizes, block_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(block_sizes[0], block_sizes[0], n=depths[0],
                        activation=self.activation, block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n,
                          activation=self.activation, block=block, *args, **kwargs)
                          for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
    
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    
class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, block_sizes=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2], activation='relu', block=ResNetBasicBlock,
                 *args, **kwargs):
        super().__init__()
        self.block_sizes = block_sizes
        self.activation=activation
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(block_sizes, block_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(block_sizes[0], block_sizes[0], n=depths[0],
                        activation=self.activation, block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n,
                          activation=self.activation, block=block, *args, **kwargs)
                          for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
    
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
        
class ResNetDecoder(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)
    
    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, maxpool, *args, **kwargs):
        super(ResNet, self).__init__()
        if maxpool:
            self.encoder = ResNetEncoder(in_channels, *args, **kwargs)  
        else:
            self.encoder = ResNetEncoderNoMaxPool(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
def resnet(in_channels, n_classes, block=ResNetBottleNeckBlock):
    return ResNet(in_channels, n_classes, maxpool=True, block=block, depths=[1, 1, 1, 1])

def resnet18(in_channels, n_classes, block=ResNetBasicBlock):
    return ResNet(in_channels, n_classes, maxpool=True, block=block, deepths=[2, 2, 2, 2])

def resnet50(in_channels, n_classes, block=ResNetBottleNeckBlock):
    return ResNet(in_channels, n_classes, maxpool=True, block=block, deepths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes, block=ResNetBottleNeckBlock):
    return ResNet(in_channels, n_classes, maxpool=True, block=block, depths=[3, 4, 23, 3])

def resnet151(in_channels, n_classes, block=ResNetBottleNeckBlock):
    return ResNet(in_channels, n_classes, maxpool=True, block=block, depths=[3, 8, 36, 3])