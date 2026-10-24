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
        super(Connect4Net, self).__init__()  # Ensure correct inheritance
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_dim, 128, kernel_size=4, stride=1, padding=2),  # Use state_dim for flexibility
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 9 * 10, 1024),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(1024, action_dim)
        self.value_head = nn.Linear(1024, 1)

    def forward(self, x):
        assert x.shape[1] == 2, f"Expected input with 2 channels, got {x.shape[1]} channels."
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        log_policy = F.log_softmax(self.policy_head(x), dim=1)  # Ensure log_softmax is applied
        value = torch.tanh(self.value_head(x))
        return log_policy, value

# Ensure no duplicated classes or functions that are now in agent_code.py