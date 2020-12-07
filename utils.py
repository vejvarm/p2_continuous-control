import torch
from torch import nn
from torch.nn import functional as F


# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
