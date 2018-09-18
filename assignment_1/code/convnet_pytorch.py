"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        super(ConvNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.AvgPool2d(kernel_size=1, stride=1, padding=0))
        self.layers.append(nn.Linear(512, n_classes))

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        for layer in self.layers[:-1]:
            x = layer(x)

        out = self.layers[-1](x.view(x.shape[0], -1))

        return out
