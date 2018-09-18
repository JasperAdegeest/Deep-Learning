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
import matplotlib.pyplot as plt


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '4000, 4000, 4000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 10000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

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
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    n_channels, image_width, image_height = cifar10['train'].images[0].shape
    n_inputs = n_channels * image_width * image_height
    n_outputs = len(cifar10['train'].labels[0])

    # Init MLP
    net = MLP(n_inputs, dnn_hidden_units, n_outputs)

    print(net)

    # Init optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=FLAGS.learning_rate)

    # Plot variables
    steps = []
    losses = []
    accuracies = []
    running_loss = 0.0

    for step in range(FLAGS.max_steps):
        optimizer.zero_grad()

        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        x = x.reshape(FLAGS.batch_size, -1)
        x, y = torch.tensor(x, requires_grad=True).to(device), torch.tensor(y, dtype=torch.float).to(device)

        outputs = net(x)
        labels = torch.max(y, 1)[1]
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == FLAGS.eval_freq - 1:
            # Calculate accuracy on the test set
            x_test, y_test = cifar10['test'].images, cifar10['test'].labels
            x_test = x_test.reshape(x_test.shape[0], -1)
            x_test, y_test = torch.tensor(x_test), torch.tensor(y_test, dtype=torch.float)
            test_outputs = net(x_test)

            steps.append(step)
            losses.append(running_loss / FLAGS.eval_freq)
            accuracies.append(accuracy(test_outputs, y_test))

            print('Epoch: {}\t Step: {}\t Loss: {}\t Test Accuracy: {}'
                  .format(cifar10['train'].epochs_completed + 1, step + 1,
                          running_loss / FLAGS.eval_freq, accuracies[-1]))
            running_loss = 0.0

    print('Finished Training')

    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, '.-')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(steps, accuracies, '.-')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')

    plt.show()


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