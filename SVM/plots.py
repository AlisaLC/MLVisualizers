import matplotlib.pyplot as plt
import matplotlib.colors
from kernels import *
from norms import *
from utils import fig2img

def plot_norm_kernel_2d(X, norm: Norm, kernel: Kernel, log_scale=False):
    distances = norm(X)
    K = kernel(distances)
    norm = None
    if log_scale:
        norm = matplotlib.colors.LogNorm()
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=K.numpy(), norm=norm)
    plt.colorbar()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')

def plot_kernel(kernel, norm, log_scale):
    X_1D = torch.linspace(-5, 5, 100).unsqueeze(1)
    X = torch.linspace(-5, 5, 100)
    Y = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    X_2D = torch.stack([X, Y], dim=2).view(-1, 2)
    if norm == 'manhattan':
        norm = ManhattanNorm()
    elif norm == 'euclidean':
        norm = EuclideanNorm()
    elif norm == 'maximum':
        norm = MaximumNorm()
    if kernel == 'linear':
        kernel = LinearKernel()
    elif kernel == 'quadratic':
        kernel = QuadraticKernel()
    elif kernel == 'gaussian':
        kernel = GaussianKernel()
    plot_norm_kernel_2d(X_2D, norm, kernel, log_scale=log_scale)
    return fig2img()