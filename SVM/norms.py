import torch
from torch import nn

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        raise NotImplementedError

class EuclideanNorm(Norm):
    def __init__(self):
        super(EuclideanNorm, self).__init__()

    def forward(self, x):
        return torch.sqrt(torch.sum(x**2, dim=-1))

class ManhattanNorm(Norm):
    def __init__(self):
        super(ManhattanNorm, self).__init__()

    def forward(self, x):
        return torch.sum(torch.abs(x), dim=-1)

class MaximumNorm(Norm):
    def __init__(self):
        super(MaximumNorm, self).__init__()

    def forward(self, x):
        return torch.max(torch.abs(x), dim=-1)[0]