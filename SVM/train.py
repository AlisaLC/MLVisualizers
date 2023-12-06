from torch import optim
from torch.utils.data import DataLoader
from SVM.SVM import SVMLoss
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.plot import fig2img
from utils.data import device

def plot_points_with_labels(svm, x, y_true, y_pred):
    y_pred = np.sign(y_pred)
    c = np.where(y_true == 1.0, 'g', 'b')
    c = np.where(y_pred == y_true, c, 'r')
    plt.scatter(x[:, 0], x[:, 1], c=c)
    X_min = x[:, 0].min()
    X_max = x[:, 0].max()
    y_min = x[:, 1].min()
    y_max = x[:, 1].max()
    XX, YY = np.mgrid[X_min:X_max:200j, y_min:y_max:200j]
    Z = svm(torch.from_numpy(np.c_[XX.ravel(), YY.ravel()]).float().to(device)).squeeze().detach().cpu().numpy()
    Z_contour = Z.copy()
    Z_contour[Z_contour > 1.0] = 4.0
    Z_contour[(Z_contour > 0.0) & (Z_contour <= 1.0)] = 3.0
    Z_contour[(Z_contour > -1.0) & (Z_contour <= 0.0)] = 2.0
    Z_contour[Z_contour <= -1.0] = 1.0
    Z_contour = Z_contour.reshape(XX.shape)
    plt.contourf(XX, YY, Z_contour, cmap=plt.cm.Paired, alpha=0.2)
    plt.contour(XX, YY, Z_contour, colors='k', linewidths=0.5)
    plt.xlim(X_min, X_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')

def train_svm(svm, X, y, progress, epochs=200, lr=0.01, C=1.0):
    loader = DataLoader(list(zip(X, y)), batch_size=128, shuffle=True)
    optimizer = optim.Adam(svm.parameters(), lr=lr)
    criterion = SVMLoss(C=C)
    for epoch in progress.tqdm(range(epochs), desc='Training SVM'):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            X = X.view(X.size(0), -1)
            y_pred = svm(X).squeeze()
            loss = criterion(y_pred, y.float(), svm.linear.weight)
            loss.backward()
            optimizer.step()
    x_s = []
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for X, y in loader:
            x_s.extend(X.tolist())
            y_trues.extend(y.tolist())
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1)
            y_pred = svm(X).squeeze()
            y_preds.extend(y_pred.detach().cpu().tolist())
        plot_points_with_labels(svm, np.array(x_s), np.array(y_trues), np.array(y_preds))
    return fig2img()
        