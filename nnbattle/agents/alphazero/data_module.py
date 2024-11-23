# nnbattle/agents/alphazero/data_module.py

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO


class ConnectFourDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, mcts_prob, reward = self.data[idx]
        # Ensure state has the correct shape [2, 6, 7]
        if isinstance(state, np.ndarray):
            # Handle if state comes in wrong shape
            if state.shape[0] != 2:
                if len(state.shape) == 2:  # If it's [6, 7]
                    state = np.stack([state == 1, state == 2])
                elif state.shape[0] > 2:  # If it has too many channels
                    state = state[:2]
            
            # Ensure state has shape [2, 6, 7]
            assert state.shape == (2, 6, 7), f"Invalid state shape: {state.shape}"
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(mcts_prob, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32)
        )


class ConnectFourDataModule(pl.LightningDataModule):
    def __init__(self, agent, num_games):
        super().__init__()
        self.agent = agent
        self.num_games = num_games
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