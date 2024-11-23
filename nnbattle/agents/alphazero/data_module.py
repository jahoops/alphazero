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
        return torch.tensor(state, dtype=torch.float32), torch.tensor(mcts_prob, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32)


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
        return DataLoader(self.dataset, batch_size=64, shuffle=True)