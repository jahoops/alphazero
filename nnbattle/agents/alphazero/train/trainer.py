# nnbattle/agents/alphazero/train/train_alpha_zero.py

import os
import time
from datetime import timedelta
import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent
from nnbattle.game import ConnectFourGame
from nnbattle.agents.alphazero.agent_code import initialize_agent  # Moved import here
from nnbattle.agents.alphazero.data_module import ConnectFourDataModule
from nnbattle.agents.alphazero.lightning_module import ConnectFourLightningModule
from nnbattle.agents.alphazero.utils.model_utils import (
    MODEL_PATH,
    load_agent_model,
    save_agent_model
)

# Configure logging at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set float32 matmul precision for Tensor Cores
torch.set_float32_matmul_precision('high')

def log_gpu_info(agent):
    """Log GPU information."""
    if torch.cuda.is_available():
        logger.info(f"Training on GPU: {torch.cuda.get_device_name()}")
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        logger.warning("No CUDA device available")

def self_play(agent, num_games):
    memory = []
    game = ConnectFourGame()
    for game_num in range(num_games):
        game.reset()
        agent.team = 1  # Initialize current player at the start of the game
        logger.info(f"Starting game {game_num + 1}/{num_games}")
        game_start_time = time.time()
        while not game.is_terminal():
            # Unpack selected_action and action_probs
            selected_action, action_probs = agent.select_move(game)
            game.make_move(selected_action, agent.team)  # Pass only the action, not the tuple
            agent.team = 3 - agent.team  # Update current player
        game_end_time = time.time()
        logger.info(f"Time taken for game {game_num + 1}: {game_end_time - game_start_time:.4f} seconds")
        result = game.get_result()
        # Ensure that state is preprocessed correctly
        preprocessed_state = agent.preprocess(game.get_state())  # Shape: [2,6,7]
        mcts_prob = torch.zeros(agent.action_dim, dtype=torch.float32)  # Initialize as Tensor
        memory.append((preprocessed_state, mcts_prob, result))
        logger.info(f"Finished game {game_num + 1}/{num_games} with result: {result}")
    agent.memory.extend(memory)  # Assuming agent.memory is a list
    return memory

# Ensure train_alphazero is defined here and not imported from elsewhere
# Do not add any import statements for train_alphazero here

def train_alphazero(
    max_iterations: int,  # Renamed from time_limit to max_iterations
    num_self_play_games: int,
    use_gpu: bool,
    load_model: bool
):
    """
    Trains the AlphaZero agent using self-play and reinforcement learning.

    :param max_iterations: Maximum number of training iterations.
    :param num_self_play_games: Number of self-play games per training iteration.
    :param use_gpu: Whether to use GPU for training.
    :param load_model: Whether to load an existing model before training.
    """
    agent = initialize_agent(
        action_dim=7,
        state_dim=2,
        use_gpu=use_gpu,
        num_simulations=800,
        c_puct=1.4,
        load_model=load_model
    )
    
    data_module = ConnectFourDataModule(agent, num_self_play_games)
    lightning_module = ConnectFourLightningModule(agent)
    
    trainer = pl.Trainer(
        # Removed max_time parameter
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,  # Set logging interval to 1
        fast_dev_run=False  # Ensure fast_dev_run is False for actual training
    )
    
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Starting training iteration {iteration}/{max_iterations}...")
        logger.info("Starting self-play games...")
        data_module.generate_self_play_games()
        
        logger.info("Starting training iteration...")
        trainer.fit(lightning_module, data_module)
        
        logger.info("Saving the model...")
        save_agent_model(agent)

        logger.info(f"Completed training iteration {iteration}/{max_iterations}.")
    
    logger.info("Training completed.")

if __name__ == "__main__":
    # Ensure CUDA_VISIBLE_DEVICES is set
    train_alphazero(
        max_iterations=10,  # Replaced time_limit with max_iterations
        num_self_play_games=2,
        use_gpu=True,
        load_model=False
    )