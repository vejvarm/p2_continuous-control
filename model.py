import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_size(in_size, ks, strd):
    """ Calculate output size of conv1d layer """
    return (in_size - ks + strd)//strd


class View(nn.Module):

    def __init__(self, full_shape):
        super(View, self).__init__()
        self.shape = full_shape

    def forward(self, x):
        return x.view(self.shape)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, n_filt=(8, 16, 32), kernel_size=(9, 5, 3), stride=(2, 2, 1), fc_units=(64, 32)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (33)
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """

        """
        INPUT: tmax*nparal (batch_size) x state_size (33)
        OUTPUT: tmax*nparal (batch_size) x action_size (4)
        
        conv output_size=(input_size - kernel_size + stride)/stride (rounded-down)
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()  # container for layer objects

        # Convolutional layers
        out_size = state_size
        nf_old = 1
        for nf, ks, strd in zip(n_filt, kernel_size, stride):
            self.layers.append(nn.Conv1d(nf_old, nf, kernel_size=ks, stride=strd))
            out_size = conv_out_size(out_size, ks, strd)
            nf_old = nf
        self.flat_size = n_filt[-1]*out_size  # 96

        # view layer for flattening
        self.layers.append(View((-1, self.flat_size)))

        # Feed-Forward layers
        self.layers.append(nn.Linear(self.flat_size, fc_units[0]))  # first fc layer
        for i in range(1, len(fc_units)):
            self.layers.append(nn.Linear(fc_units[i-1], fc_units[i]))  # middle fc layers
        self.layers.append(nn.Linear(fc_units[-1], action_size))  # last fc layer

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state.unsqueeze(1)  # add dimension for the convolutions
        x = F.relu(self.layers[0](x))      # first layer
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x))               # middle layers
        return torch.tanh(self.layers[-1](x))  # last layer


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
        input_state shape: tmax*nparal (batch_size) x state_size
        action shape: tmax*nparal (batch_size) x action_size
        """
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
