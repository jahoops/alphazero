# nnbattle/agents/alphazero/data_module.py

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO

# Ensure __all__ is defined
__all__ = ['ConnectFourDataset', 'ConnectFourDataModule']

class ConnectFourDataset(Dataset):
    def __init__(self, data):
        self.data = data if data else []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, mcts_prob, reward = self.data[idx]
        # Ensure state has the correct shape [2, 6, 7]
        if isinstance(state, np.ndarray):
            if state.shape != (2, 6, 7):
                logger.error(f"Invalid state shape: {state.shape}. Expected (2, 6, 7).")
                raise ValueError(f"Invalid state shape: {state.shape}. Expected (2, 6, 7).")
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(mcts_prob, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32)
        )


class ConnectFourDataModule(pl.LightningDataModule):
    def __init__(self, agent, num_games):
        super().__init__()
        self.agent = agent
        self.num_games = max(1, num_games)  # Ensure at least one game
        self.dataset = ConnectFourDataset([])

    def generate_self_play_games(self):
        for _ in range(self.num_games):
            self.agent.self_play()
        self.dataset.data.extend(self.agent.memory)
        self.agent.memory.clear()
        logger.info(f"Generated {self.num_games} self-play games.")

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        if len(self.dataset.data) == 0:
            # Add at least one dummy sample if empty
            self.dataset.data.append((
                np.zeros((2, 6, 7)),  # state
                np.zeros(7),          # mcts_probs
                0                     # reward
            ))
        return DataLoader(self.dataset, batch_size=64, shuffle=True)