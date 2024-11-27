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
        # Initial convolution
        self.conv1 = nn.Conv2d(state_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, action_dim)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 6 * 7, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Ensure device attribute is set
        self._device = torch.device('cpu')

        # Set appropriate default modes for different components
        self.train()  # Set default mode to train
        self.bn1.eval()  # BatchNorm layers should typically be in eval during inference
        self.policy_bn.eval()
        self.value_bn.eval()
        
        # Set dropout if used (for example)
        self.training = True  # Explicitly set training mode

    def to(self, device):
        super().to(device)
        self._device = device
        return self  # Ensure to return self for chaining

    def forward(self, x):
        # Ensure batch norm layers are in correct mode during forward pass
        self.bn1.train(self.training)
        self.policy_bn.train(self.training)
        self.value_bn.train(self.training)
        
        # Ensure x is on the correct device
        x = x.to(self._device)

        # Initial convolutional block
        x = self.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.res_blocks(x)

        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def train(self, mode=True):
        """Override train method to handle batch norm layers correctly."""
        super().train(mode)
        if not mode:
            # When setting to eval mode, ensure batch norms are properly configured
            self.bn1.eval()
            self.policy_bn.eval()
            self.value_bn.eval()
        return self

__all__ = ['Connect4Net']