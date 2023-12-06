import torch
from torch import nn
from SVM.RBF import RBF

class SVMLoss(nn.Module):
    def __init__(self, C=1.0):
        super(SVMLoss, self).__init__()
        self.C = C
    
    def forward(self, y_pred, y_true, W):
        return torch.sum(W**2) / 2. + self.C * torch.mean(torch.clamp(1 - y_pred * y_true, min=0))

class SVM(nn.Module):
    def __init__(self, input_dim: int):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, X):
        return self.linear(X)

class KernelSVM(SVM):
    def __init__(self, input_dim: int, hidden_dim: int, kernel, norm):
        super(KernelSVM, self).__init__(hidden_dim)
        self.kernel = RBF(input_dim, hidden_dim, kernel, norm)

    def forward(self, X):
        features = self.kernel(X)
        return self.linear(features)