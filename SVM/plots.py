import matplotlib.pyplot as plt
import matplotlib.colors
from kernels import *
from norms import *
from utils.plot import fig2img
from utils.data import generate_2D

norm_dict = {
    'manhattan': ManhattanNorm(),
    'euclidean': EuclideanNorm(),
    'maximum': MaximumNorm()
}

kernel_dict = {
    'linear': LinearKernel(),
    'quadratic': QuadraticKernel(),
    'gaussian': GaussianKernel()
}

def plot_norm_kernel_2d(X, norm: Norm, kernel: Kernel, log_scale=False):
    distances = norm(X)
    K = kernel(distances)
    plt.scatter(
        X[:, 0].numpy(),
        X[:, 1].numpy(),
        c=K.numpy(),
        norm=matplotlib.colors.LogNorm() if log_scale else None
    )
    plt.colorbar()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')

def plot_kernel(kernel, norm, log_scale):
    X_2D = generate_2D([-5, 5], [-5, 5])
    norm = norm_dict[norm]
    kernel = kernel_dict[kernel]
    plot_norm_kernel_2d(X_2D, norm, kernel, log_scale=log_scale)
    return fig2img()