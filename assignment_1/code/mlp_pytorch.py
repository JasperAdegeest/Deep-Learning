"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        super(MLP, self).__init__()

        self.linear_layers = []
        prev_n_units = n_inputs
        for n_units in n_hidden:
            if len(self.linear_layers) == 0:
                layer = nn.Linear(n_inputs, n_units)
            else:
                layer = nn.Linear(prev_n_units, n_units)

            prev_n_units = n_units
            self.linear_layers.append(layer)

        self.final_layer = nn.Linear(prev_n_units, n_classes)

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

        for layer in self.linear_layers:
            x = layer(x)
            x = F.relu(x)

        out = self.final_layer(x)

        return out
