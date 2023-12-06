def generate_1D(X_range, points=100):
    return torch.linspace(X_range[0], X_range[1], points).view(-1, 1)

def generate_2D(X_range, Y_range, points=100):
    X = torch.linspace(X_range[0], X_range[1], points)
    Y = torch.linspace(Y_range[0], Y_range[1], points)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    return torch.stack([X, Y], dim=2).view(-1, 2)