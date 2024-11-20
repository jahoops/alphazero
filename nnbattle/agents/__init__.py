# /home/j/GIT/nnbattle/agents/__init__.py

from .alphazero import AlphaZeroAgent, Connect4LightningModule, ConnectFourDataModule
from .minimax import MinimaxAgent

__all__ = [
    'AlphaZeroAgent',
    'Connect4LightningModule',
    'ConnectFourDataModule',
    'MinimaxAgent',
]