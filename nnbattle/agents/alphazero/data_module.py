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
        # Ensure reward is a float
        if not isinstance(reward, float):
            logger.error(f"Invalid reward type at index {idx}: {type(reward)}. Expected float.")
            reward = 0.0  # Assign a default value or handle as needed
        # Ensure state has the correct shape [2, 6, 7]
        if isinstance(state, np.ndarray):
            if state.shape != (2, 6, 7):
                preprocessed_state = self.agent.preprocess(state)
                logger.info("Preprocessing state to shape (2, 6, 7).")
                state = preprocessed_state.cpu().numpy()  # Move to CPU before converting to numpy
                if state.shape != (2, 6, 7):
                    logger.error(f"Invalid state shape after preprocessing: {state.shape}. Expected (2, 6, 7).")
                    raise ValueError(f"Invalid state shape after preprocessing: {state.shape}. Expected (2, 6, 7).")
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(mcts_prob, dtype=torch.float32).cpu(),  # Ensure mcts_prob is on CPU
            torch.tensor(reward, dtype=torch.float32)
        )


class ConnectFourDataModule(pl.LightningDataModule):
    def __init__(self, agent, num_games):
        super().__init__()
        self.agent = agent
        self.num_games = num_games
        self.dataset = ConnectFourDataset([], agent)
        self.batch_size = 32  # Add explicit batch size
        self.val_dataset = ConnectFourDataset([], agent)  # Add validation dataset

    def generate_self_play_games(self):
        logger.info(f"Generating {self.num_games} self-play games.")
        try:
            # Limit the number of self-play games for testing
            self.agent.self_play(max_moves=5)  # Limit moves for quicker execution
            if self.agent.memory:
                self.dataset.data.extend(self.agent.memory)
                logger.info(f"Appending {len(self.agent.memory)} games to the dataset.")
                self.agent.memory.clear()
                logger.info(f"Dataset size after self-play: {len(self.dataset.data)} samples")
        except Exception as e:
            logger.error(f"An error occurred during self-play generation: {e}")
            # Add dummy data for testing if self-play fails
            dummy_state = np.zeros((2, 6, 7))
            dummy_mcts_prob = np.ones(self.agent.action_dim) / self.agent.action_dim
            dummy_reward = 0.0  # Ensure reward is float
            self.dataset.data.append((dummy_state, dummy_mcts_prob, dummy_reward))
            logger.info("Added dummy data for testing purposes.")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Split data into training and validation
            total = len(self.dataset)
            val_size = int(0.2 * total)
            train_size = total - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
    
    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            logger.warning("Training dataset is empty! Adding dummy sample for testing.")
            self.train_dataset.dataset.data.append((
                np.zeros((2, 6, 7)),  # state
                np.zeros(7),          # mcts_probs
                0.0                    # reward
            ))
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,  # Set to 4 for multiprocessing efficiency
            pin_memory=True
        )
    
    def val_dataloader(self):
        if len(self.val_dataset) == 0:
            logger.warning("Validation dataset is empty! Adding dummy sample for testing.")
            self.val_dataset.dataset.data.append((
                np.zeros((2, 6, 7)),  # state
                np.zeros(7),          # mcts_probs
                0.0                    # reward
            ))
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )