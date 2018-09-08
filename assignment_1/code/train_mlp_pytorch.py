"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch.optim as optim
import torch.nn as nn
import torch


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '500, 500, 500, 500, 500, 500'
LEARNING_RATE_DEFAULT = 5e-3
MAX_STEPS_DEFAULT = 3000
BATCH_SIZE_DEFAULT = 50
EVAL_FREQ_DEFAULT = 500

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    prediction_classes = torch.max(predictions, 1)[1]
    target_classes = torch.max(targets, 1)[1]

    if len(prediction_classes) != len(target_classes):
        raise ValueError('Predictions and targets are not of the same size')

    correct = 0
    for i, prediction in enumerate(prediction_classes):
        if prediction == target_classes[i]:
            correct += 1

    accuracy = correct / len(prediction_classes)

    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
      dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
      dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
      dnn_hidden_units = []

    # Init device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get data and its properties
    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
    n_channels, image_width, image_height = cifar10['train'].images[0].shape
    n_inputs = n_channels * image_width * image_height
    n_outputs = len(cifar10['train'].labels[0])

    # Init MLP
    net = MLP(n_inputs, dnn_hidden_units, n_outputs)
    net.to(device)

    # Init optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
    running_loss = 0.0

    for step in range(MAX_STEPS_DEFAULT):
        x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
        x = x.reshape(BATCH_SIZE_DEFAULT, -1)
        x, y = torch.tensor(x, requires_grad=True), torch.tensor(y, dtype=torch.int64)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = net(x)
        labels = torch.max(y, 1)[1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % EVAL_FREQ_DEFAULT == EVAL_FREQ_DEFAULT - 1:
            net.eval()

            # Calculate accuracy on the test set
            x_test, y_test = cifar10['test'].images, cifar10['test'].labels
            x_test = x_test.reshape(x_test.shape[0], -1)
            x_test, y_test = torch.tensor(x_test), torch.tensor(y_test, dtype=torch.int64)
            test_outputs = net(x_test)

            print('Epoch: {}\t Step: {}\t Loss: {}\t Test Accuracy: {}'
                  .format(cifar10['train'].epochs_completed + 1, step + 1,
                          running_loss / EVAL_FREQ_DEFAULT, accuracy(test_outputs, y_test)))
            running_loss = 0.0

            net.train()

            # print('Epoch: {}\t Step: {}\t Loss: {}\t Accuracy: {}'
            #       .format(cifar10['train'].epochs_completed + 1, step + 1,
            #               running_loss / EVAL_FREQ_DEFAULT, accuracy(outputs, y)))
            # running_loss = 0.0

    print('Finished Training')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
      print(key + ' : ' + str(value))

def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
      os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()