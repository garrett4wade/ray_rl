import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return x + self.net(x)


class ConvMaxpoolResModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, 2 * c_in, kernel_size=3, stride=1, padding=1),
                                 nn.MaxPool2d(kernel_size=2, stride=2), ResidualBlock(2 * c_in),
                                 ResidualBlock(2 * c_in))

    def forward(self, x):
        return self.net(x)
