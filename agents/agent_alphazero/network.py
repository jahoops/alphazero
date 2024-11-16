# alphazero_agent/network.py

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
    def __init__(self, state_dim, action_dim, num_res_blocks=5, num_filters=128):
        super(Connect4Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=4, padding=2, stride=2),  # Output: (num_filters, 4, 4)
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, padding=2, stride=2),  # (num_filters*2, 2, 2)
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=4, padding=2, stride=2),  # (num_filters*2, 1,1)
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters * 2) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters * 2, 2, kernel_size=1),  # Output: (2, 1, 1)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (batch_size, 2)
            nn.Linear(8, action_dim),  # Updated in_features from 2 to 8
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters * 2, 1, kernel_size=1),  # Output: (1, 1, 1)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (batch_size, 4)
            nn.Linear(4, 256),  # Updated in_features from 1 to 4
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return F.log_softmax(policy, dim=1), value