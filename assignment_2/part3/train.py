# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################


def train(config):
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden, device=device,
                                dropout=1-config.dropout_keep_prob).to(device)

    if os.path.exists(config.save_file):
        print("Load model: {}".format(config.save_file))
        model.load_state_dict(torch.load(config.save_file, map_location=lambda storage, loc: storage))

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.1, patience=10)

    nr_of_epochs = round(config.train_steps / len(data_loader))

    for epoch in range(nr_of_epochs):
        epoch_loss = 0
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = torch.stack(batch_inputs)
            batch_size = batch_inputs.shape[1]
            batch_inputs = batch_inputs.view(config.seq_length, batch_size, 1)

            batch_inputs = batch_inputs.type(torch.float)
            batch_targets = torch.stack(batch_targets).to(device)

            # Only for time measurement of step through network
            t1 = time.time()

            optimizer.zero_grad()

            h0 = Variable(torch.zeros(config.lstm_num_layers, batch_size, config.lstm_num_hidden).to(device))
            c0 = Variable(torch.zeros(config.lstm_num_layers, batch_size, config.lstm_num_hidden).to(device))
            outputs, _, _ = model.forward(batch_inputs, h0, c0)

            losses = []
            accuracies = []
            for i, output in enumerate(outputs):
                target = batch_targets[i, :]
                loss = criterion(output, target)
                losses.append(loss)

                predictions = torch.max(output, 1)[1]
                accuracies.append(torch.sum(predictions.eq(target)).item() / len(target))

            loss = sum(losses) / len(losses)
            epoch_loss += loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            optimizer.step()

            loss = loss.item()
            accuracy = sum(accuracies) / len(accuracies)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Epoch {:02}, Step {:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch+1, step, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                model.eval()

                start_tokens = [random.randint(1,dataset.vocab_size-1) for _ in range(10)]
                summary_file = open("{}{}_{}_summaries.txt".format(config.summary_path, epoch, step), 'w')
                for start_token in start_tokens:
                    sentence = [start_token]
                    next_char = start_token
                    h0 = Variable(torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden).to(device))
                    c0 = Variable(torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden).to(device))
                    for i in range(config.seq_length):
                        input_batch = torch.stack([torch.tensor([next_char], dtype=torch.float)])
                        input_batch = input_batch.view(1, 1, 1)
                        input_batch = input_batch.to(device)

                        output, h0, c0 = model(input_batch, h0, c0)
                        output = F.softmax(output[-1, :, :]).detach().cpu().numpy()
                        output = output[0]
                        output = (np.log(output) / config.temperature)
                        output = np.exp(output) / np.sum(np.exp(output))
                        next_char = np.argmax(np.random.multinomial(1, output, 1))
                        sentence.append(next_char)

                    char_sentence = dataset.convert_to_string(sentence)
                    summary_file.write(''.join(char_sentence) + '\n')
                summary_file.close()

                model.train()

                # Save model
                torch.save(model.state_dict(), config.save_file)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        scheduler.step(epoch_loss)

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='book.txt', help="Path to a .txt file to train on")
    parser.add_argument('--save_file', type=str, default='model.pt', help="Path to a .txt file to train on")
    parser.add_argument('--temperature', type=float, default=1.0)


    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.6, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=2e5, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=20, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=20, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
