# /network.py

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Ensure no in-place operations
        out = self.relu(out)
        return out

class Connect4Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Connect4Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 9, 1024),  # Adjusted dimensions
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x