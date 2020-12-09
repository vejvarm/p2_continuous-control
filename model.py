# heavily inspired by https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_size(in_size, ks, strd):
    """ Calculate output size of a 1D convolutional layer
    Params
    ======
        in_size (int): input length
        ks (int): kernel size
        strd (int): stride of kernel over the input
    """
    return (in_size - ks + strd)//strd


class View(nn.Module):
    """ A simple View layer module for reshaping a tensor """
    def __init__(self, full_shape):
        super(View, self).__init__()
        self.shape = full_shape

    def forward(self, x):
        return x.view(self.shape)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, n_filt=(8, 16, 32), kernel_size=(9, 5, 3), stride=(2, 2, 1), fc_units=(128, 64, 32)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (33)
            action_size (int): Dimension of each action
            seed (int): Random seed
            n_filt (Tuple[int]): number of filters (kernels) in each conv1d layer
            kernel_size (Tuple[int]): size of kernels in each conv1d layer
            stride (Tuple[int]): stride step of kernels over the inputs for each conv1d layer
            fc_units (Tuple[int]): Number of hidden nodes in fully connected layers
        """

        """
        INPUT: tmax*nparal (batch_size) x state_size (33)
        OUTPUT: tmax*nparal (batch_size) x action_size (4)
        
        
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()  # container for layer objects

        # Convolutional layers
        out_size = state_size
        nf_old = 1
        for nf, ks, strd in zip(n_filt, kernel_size, stride):
            self.layers.append(nn.Conv1d(nf_old, nf, kernel_size=ks, stride=strd))  # conv1d layers with BN
            self.layers.append(nn.BatchNorm1d(nf))
            out_size = conv_out_size(out_size, ks, strd)
            nf_old = nf
        self.flat_size = n_filt[-1]*out_size  # calculate final flattened output size of conv layers

        # View layer for flattening
        self.layers.append(View((-1, self.flat_size)))

        # Feed-Forward (fully-connected) layers
        self.layers.append(nn.Linear(self.flat_size, fc_units[0]))     # first fc layer with BN
        self.layers.append(nn.BatchNorm1d(fc_units[0]))
        for i in range(1, len(fc_units)):
            self.layers.append(nn.Linear(fc_units[i-1], fc_units[i]))  # middle fc layers with BN
            self.layers.append(nn.BatchNorm1d(fc_units[i]))
        self.layers.append(nn.Linear(fc_units[-1], action_size))       # last fc layer

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state.unsqueeze(1)                  # add dimension for the convolutions
        x = F.relu(self.layers[0](x))           # first layer
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x))                # middle layers
        return torch.tanh(self.layers[-1](x))   # last layer


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
