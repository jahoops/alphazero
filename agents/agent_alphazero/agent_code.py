# alphazero_agent/agent_code.py

import logging
import math
import numpy as np
import random
from collections import deque
from copy import deepcopy

import torch

from .network import Connect4Net
from .mcts import MCTSNode
from .utils import preprocess_board  # Assuming you have utility functions
from connect_four_game import ConnectFourGame  # Update the import path as needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AlphaZeroAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        use_gpu=False,
        model_path="alphazero_agent/model/alphazero_model_final.pth",
        num_simulations=800,
        c_puct=1.4
    ):
        """
        Initializes the AlphaZeroAgent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model = Connect4Net(state_dim, action_dim).to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        self.env = ConnectFourGame()  # Initialize the game environment/state
        self.model_path = model_path
        self.model_loaded = False  # Flag to check if model is loaded
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Memory for storing training samples
        self.memory = deque(maxlen=10000)  # Use deque for efficient FIFO operations

        # Initialize current_player
        self.current_player = 1  # Player 1 starts by default

    # ... (Other methods remain unchanged)

    def train_step(self, batch_size=32):
        """
        This method can be deprecated or refactored to work with LightningModule.
        """
        pass  # Training is now handled by PyTorch Lightning