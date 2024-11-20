# /data_module.py

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ConnectFourDataset(Dataset):
    def __init__(self, memory):
        self.memory = memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        state, mcts_prob, value = self.memory[idx]
        return torch.FloatTensor(state), torch.FloatTensor(mcts_prob), torch.FloatTensor([value])

class ConnectFourDataModule(pl.LightningDataModule):
    def __init__(self, agent, batch_size=32):
        super().__init__()
        self.agent = agent
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ConnectFourDataset(self.agent.memory)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)