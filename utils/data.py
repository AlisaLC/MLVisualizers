import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def generate_1D(X_range, points=100):
    return torch.linspace(X_range[0], X_range[1], points).view(-1, 1)

def generate_2D(X_range, Y_range, points=100):
    X = torch.linspace(X_range[0], X_range[1], points)
    Y = torch.linspace(Y_range[0], Y_range[1], points)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    return torch.stack([X, Y], dim=2).view(-1, 2)

def generate_standard_moons(n_samples=1000, noise=0.1, random_state=None):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=None)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y[y == 0] = -1
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y