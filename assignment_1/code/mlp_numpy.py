"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *
import numpy as np

class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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

        self.layers = []
        prev_n_units = n_inputs
        for n_units in n_hidden:
            if len(self.layers) == 0:
                self.layers.append(LinearLayer(n_inputs, n_units))
            else:
                self.layers.append(LinearLayer(prev_n_units, n_units))

            prev_n_units = n_units

        self.layers.append(LinearLayer(prev_n_units, n_classes))

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

        relu = ReLU()

        for i, layer in enumerate(self.layers):
            if i != (len(self.layers) - 1):
                x = layer.forward(x)
                x = relu.forward(x)

        out = self.layers[-1].forward(x)

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return


class LinearLayer:
    def __init__(self, n_inputs, n_units):
        self.n_inputs = n_inputs
        self.n_units = n_units

        # Create weight, bias matrices
        self.W = np.random.random((n_units, n_inputs))
        self.b = np.random.random((n_units, 1))

    def forward(self, x):
        return np.matmul(self.W, x) + self.b


class ReLU:
    def forward(self, x):
        return x.clip(min=0)

class SoftMax:
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

class CrossEntropy:
    def forward(self, x, t):
        max_t_index = np.argmax(t, axis=1)
        x_arg_max = x[max_t_index, range(len(max_t_index))]
        return -np.log(x_arg_max)


net = MLP(2, [3], 3)
criterion = CrossEntropy()

outputs = net.forward([[-4, 2], [2, 1]])
labels = [[0, 0, 1], [0, 1, 0]]

loss = criterion.forward(outputs, labels)
print(loss)



