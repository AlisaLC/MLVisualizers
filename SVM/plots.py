import matplotlib.pyplot as plt
import matplotlib.colors
from SVM.kernels import *
from SVM.norms import *
from SVM.SVM import SVM, KernelSVM
from utils.plot import fig2img
from utils.data import generate_2D, generate_standard_moons
import gradio as gr
from SVM.train import train_svm

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

def plot_norm_kernel_2d(X, norm, kernel, log_scale=False):
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

def plot_SVM(C, kernel, norm, progress=gr.Progress()):
    X, y = generate_standard_moons()
    if kernel == 'none':
        svm = SVM(2)
    else:
        svm = KernelSVM(2, 1, kernel_dict[kernel], norm_dict[norm])
    return train_svm(svm, X, y, progress, C=C)