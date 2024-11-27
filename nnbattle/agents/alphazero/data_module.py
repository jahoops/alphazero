import pytorch_lightning as pl
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from nnbattle.agents.alphazero.self_play import SelfPlay  # Updated import
from nnbattle.game.connect_four_game import ConnectFourGame

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Changed from DEBUG to INFO

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
        # Include team information in the state if not already present
        # Ensure reward is a float
        if not isinstance(reward, float):
            logger.error(f"Invalid reward type at index {idx}: {type(reward)}. Expected float.")
            reward = 0.0  # Assign a default value or handle as needed
        # Ensure state has the correct shape [3, 6, 7]
        if isinstance(state, np.ndarray):
            if state.shape != (3, 6, 7):
                preprocessed_state = self.agent.preprocess(state, self.agent.team)
                logger.info("Preprocessing state to shape (3, 6, 7).")
                state = preprocessed_state.cpu().numpy()  # Move to CPU before converting to numpy
                if state.shape != (3, 6, 7):
                    logger.error(f"Invalid state shape after preprocessing: {state.shape}. Expected (3, 6, 7).")
                    raise ValueError(f"Invalid state shape after preprocessing: {state.shape}. Expected (3, 6, 7).")
        
        # Always return CPU tensors from dataset
        if torch.is_tensor(state) and state.device.type == 'cuda':
            state = state.cpu()
        if torch.is_tensor(mcts_prob) and mcts_prob.device.type == 'cuda':
            mcts_prob = mcts_prob.cpu()
        
        # Convert to tensors on CPU
        return (
            torch.tensor(state, dtype=torch.float32),  # Remove device
            torch.tensor(mcts_prob, dtype=torch.float32),  # Remove device
            torch.tensor(reward, dtype=torch.float32)  # Remove device
        )


class ConnectFourDataModule(pl.LightningDataModule):
    def __init__(self, agent, num_games, batch_size=32, num_workers=2, persistent_workers=True):
        super().__init__()
        self.agent = agent
        self.num_games = num_games
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.dataset = ConnectFourDataset([], agent)

    def generate_self_play_games(self, temperature=1.0):
        """Generate self-play games using the SelfPlay class."""
        logger.info(f"Generating {self.num_games} self-play games with temperature {temperature}.")
        try:
            # Create a new game instance instead of passing the agent
            game = ConnectFourGame()
            self_play = SelfPlay(game=game, model=self.agent.model, num_simulations=self.agent.num_simulations)
            training_data = self_play.generate_training_data(self.num_games)
            self.dataset = ConnectFourDataset(training_data, self.agent)
            logger.info(f"Generated {len(self.dataset)} training examples.")
        except Exception as e:
            logger.error(f"An error occurred during self-play generation: {e}")
            raise

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if len(self.dataset) == 0:
                logger.warning("Dataset is empty. Assigning empty training and validation datasets.")
                self.train_dataset = torch.utils.data.Subset(self.dataset, [])
                self.val_dataset = torch.utils.data.Subset(self.dataset, [])
            else:
                # Split data into training and validation using random_split
                total = len(self.dataset)
                val_size = max(int(0.2 * total), 1)  # Ensure at least one sample
                train_size = total - val_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            logger.warning("Training dataset is empty! Training cannot proceed without data.")
            raise ValueError("Training dataset is empty. Please generate self-play games before training.")
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,  # Only pin if CUDA available
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=2,
            worker_init_fn=self._worker_init_fn
        )
    
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            logger.error("Validation dataset has not been initialized. Ensure that `setup` is called before using val_dataloader.")
            raise AttributeError("Validation dataset not initialized.")
        
        if len(self.val_dataset) == 0:
            logger.warning("Validation dataset is empty! Validation cannot proceed without data.")
            raise ValueError("Validation dataset is empty. Please generate self-play games before validation.")
        
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,  # Only pin if CUDA available
            persistent_workers=True  # Added to speed up worker initialization
        )

    def _worker_init_fn(self, worker_id: int):
        """Initialize each worker with a different seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)