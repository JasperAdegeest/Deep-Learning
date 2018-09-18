"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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

    substracted_classes = prediction_classes - target_classes
    correct = len(prediction_classes) - torch.nonzero(substracted_classes).size(0)
    accuracy = correct / len(prediction_classes)

    return accuracy

def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Get data and its properties
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    n_outputs = len(cifar10['train'].labels[0])

    # Init device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init MLP
    net = ConvNet(3, n_outputs).to(device)
    net = nn.DataParallel(net)

    print(net)

    # Init optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER_DEFAULT == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=FLAGS.learning_rate)

    # Plot variables
    steps = []
    losses = []
    accuracies = []
    running_loss = 0.0

    for step in range(FLAGS.max_steps):
        optimizer.zero_grad()

        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        x, y = torch.tensor(x, requires_grad=True).to(device), torch.tensor(y, dtype=torch.float).to(device)

        outputs = net(x)
        labels = torch.max(y, 1)[1]
        loss = criterion(outputs, labels)
        running_loss = running_loss + loss.item()
        loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == FLAGS.eval_freq - 1:
            net.eval()

            temp_accuracies = []
            # Calculate accuracy on the test set
            for i in range(len(cifar10['test'].labels)):
                x_test, y_test = cifar10['test'].next_batch(1)
                x_test, y_test = torch.tensor(x_test).to(device), torch.tensor(y_test, dtype=torch.float).to(device)
                output = net(x_test)
                temp_accuracies.append(accuracy(output.to('cpu'), y_test.to('cpu')))

            steps.append(step)
            losses.append(running_loss / FLAGS.eval_freq)
            accuracies.append(np.average(temp_accuracies))

            print('Epoch: {}\t Step: {}\t Loss: {}\t Test Accuracy: {}'
                  .format(cifar10['train'].epochs_completed + 1, step + 1,
                          running_loss / FLAGS.eval_freq, accuracies[-1]))
            running_loss = 0.0

            net.train()

    print('Finished Training')

    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, '.-')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(steps, accuracies, '.-')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')

    plt.savefig('plot.png')

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