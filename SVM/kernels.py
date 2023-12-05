import torch
from torch import nn

class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()

    def forward(self, distances):
        raise NotImplementedError

class LinearKernel(Kernel):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, distances):
        return distances

class QuadraticKernel(Kernel):
    def __init__(self):
        super(QuadraticKernel, self).__init__()

    def forward(self, distances):
        return distances.pow(2)

class GaussianKernel(Kernel):
    def __init__(self):
        super(GaussianKernel, self).__init__()

    def forward(self, distances):
        return torch.exp(-distances.pow(2))