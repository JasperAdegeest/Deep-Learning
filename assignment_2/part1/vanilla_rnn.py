################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn
from torch.autograd import Variable

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.device = device

        self.W_hx = nn.Parameter(torch.FloatTensor(input_dim, num_hidden).normal_(0, 1/num_hidden))
        self.W_hh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden).normal_(0, 1/num_hidden))
        self.b_h = nn.Parameter(torch.zeros(num_hidden,))

        self.W_ph = nn.Parameter(torch.FloatTensor(num_hidden, num_classes).normal_(0, 0.0001))
        self.b_p = nn.Parameter(torch.zeros(1, num_classes))


    def forward(self, x):
        h = Variable(torch.zeros(self.batch_size, self.num_hidden,))

        for i in range(self.seq_length):
            h = torch.mm(x[:, i].unsqueeze(1), self.W_hx) + torch.mm(h, self.W_hh) + self.b_h
            h = torch.tanh(h)
            p = torch.mm(h, self.W_ph) + self.b_p

        return p
