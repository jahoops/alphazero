# /network.py

import torch
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
        # Example architecture
        self.conv1 = nn.Conv2d(state_dim, 128, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 5, 256)
        self.fc_policy = nn.Linear(256, action_dim)
        self.fc_value = nn.Linear(256, 1)
    
    def to(self, device):
        """Ensure all parameters are moved to the specified device."""
        super().to(device)
        self._device = device
        # Explicitly move all parameters to device
        for param in self.parameters():
            param.data = param.data.to(device)
        return self

    def forward(self, x):
        # Get the device of the first parameter
        device = next(self.parameters()).device
        # Ensure input is on same device as parameters
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        policy = self.fc_policy(x)
        value = self.fc_value(x)
        return policy, value

__all__ = ['Connect4Net']