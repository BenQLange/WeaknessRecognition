import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
import random
import copy

class DRQN(nn.Module):
    def __init__(self, n_actions):
        super(DRQN, self).__init__()
        self.input_dim = 50
        self.output_dim = 50
        self.n_layers = 1
        self.n_actions = n_actions

        self.vars = nn.ParameterList()

        self.flat = Flatten()
        self.FC0 = nn.Linear(3,1)
        self.vars.append(self.FC0.weight)
        self.vars.append(self.FC0.bias)
        self.LSTM = nn.LSTM(input_size=self.input_dim, hidden_size=self.output_dim, num_layers=self.n_layers)
        self.FC1 = nn.Linear(self.output_dim, 50)
        self.vars.append(self.FC1.weight)
        self.vars.append(self.FC1.bias)
        self.FC2 = nn.Linear(50, self.n_actions)
        self.vars.append(self.FC2.weight)
        self.vars.append(self.FC2.bias)
        self.vars.append(self.LSTM.weight_ih_l0)
        self.vars.append(self.LSTM.weight_hh_l0)
        self.vars.append(self.LSTM.bias_ih_l0)
        self.vars.append(self.LSTM.bias_hh_l0)

    def forward(self, x, hidden, vars = None):

        if vars is None:
            vars = self.vars

        self.FC0.weight = vars[0]
        self.FC0.bias = vars[1]
        self.FC1.weight = vars[2]
        self.FC1.bias = vars[3]
        self.FC2.weight = vars[4]
        self.FC2.bias = vars[5]
        self.LSTM.weight_ih_l0 = vars[6]
        self.LSTM.weight_hh_l0 = vars[7]
        self.LSTM.bias_ih_l0 = vars[8]
        self.LSTM.bias_hh_l0 = vars[9]

        x = torch.from_numpy(x)
        x = x.unsqueeze(0).float()
        x = F.relu(self.FC0(x))
        x = x.squeeze(-1)
        x = x.unsqueeze(0)
        h3, new_hidden = self.LSTM(x, hidden)
        h4 = F.relu(self.FC1(h3))
        h5 = self.FC2(h4)

        return h5, new_hidden

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
