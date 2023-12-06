import torch
from torch import nn

class RBF(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, norm):
        super(RBF, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel = kernel
        self.norm = norm
        self.centers = nn.Parameter(torch.randn(self.output_dim, self.input_dim))
        self.log_sigmas = nn.Parameter(torch.randn(self.output_dim))

    def forward(self, X):
        size = (X.size(0), self.output_dim, self.input_dim)
        X = X.unsqueeze(1).expand(size)
        C = self.centers.unsqueeze(0).expand(size)
        distances = self.norm(X - C) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.kernel(distances)