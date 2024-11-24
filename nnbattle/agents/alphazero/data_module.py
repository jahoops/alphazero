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
    def __init__(self, data, agent):
        self.data = data if data else []
        self.agent = agent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, mcts_prob, reward = self.data[idx]
        # Ensure state has the correct shape [2, 6, 7]
        if isinstance(state, np.ndarray):
            if state.shape != (2, 6, 7):
                preprocessed_state = self.agent.preprocess(state)
                preprocessed_state = agent.preprocess(state)
                logger.info("Preprocessing state to shape (2, 6, 7).")
                state = preprocessed_state.numpy()
                if state.shape != (2, 6, 7):
                    logger.error(f"Invalid state shape after preprocessing: {state.shape}. Expected (2, 6, 7).")
                    raise ValueError(f"Invalid state shape after preprocessing: {state.shape}. Expected (2, 6, 7).")
        
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
        self.dataset = ConnectFourDataset([], agent)

    def generate_self_play_games(self):
        logger.info(f"Generating {self.num_games} self-play games.")
        try:
            self.agent.self_play()
            logger.info("Self-play games generated successfully.")
        except Exception as e:
            logger.error(f"An error occurred during self-play generation: {e}")
            raise

        # Ensure that agent.memory has been populated
        if not self.agent.memory:
            logger.warning("No self-play games were generated.")
        else:
            self.dataset.data.extend(self.agent.memory)
            self.agent.memory.clear()
            logger.info(f"Added {len(self.agent.memory)} games to the dataset.")

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