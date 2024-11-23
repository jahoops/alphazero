# nnbattle/agents/alphazero/train/train_alpha_zero.py

import os
import time
from datetime import timedelta
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from nnbattle.game import ConnectFourGame
from nnbattle.agents.alphazero.agent_code import initialize_agent  # Ensure correct import
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

def log_gpu_info(agent):
    """Log GPU information."""
    if torch.cuda.is_available():
        logger.info(f"Training on GPU: {torch.cuda.get_device_name()}")
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def self_play(agent, num_games):
    memory = []
    game = ConnectFourGame()
    for game_num in range(num_games):
        game.reset()
        agent.current_player = 1  # Initialize current player at the start of the game
        logger.info(f"Starting game {game_num + 1}/{num_games}")
        game_start_time = time.time()
        while not game.is_terminal():
            # Unpack selected_action and action_probs
            selected_action, action_probs = agent.select_move(game)
            game.make_move(selected_action)  # Pass only the action, not the tuple
            agent.current_player = game.current_player  # Update current player
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
    time_limit: int,
    num_self_play_games: int,
    use_gpu: bool,
    load_model: bool
):
    """
    Trains the AlphaZero agent using self-play and reinforcement learning.

    :param time_limit: Maximum time (in seconds) for training.
    :param num_self_play_games: Number of self-play games per training iteration.
    :param use_gpu: Whether to use GPU for training.
    :param load_model: Whether to load an existing model before training.
    """
    agent = AlphaZeroAgent(
        action_dim=7,
        state_dim=2,
        use_gpu=use_gpu,
        load_model=load_model
    )
    
    data_module = ConnectFourDataModule(agent, num_self_play_games)
    lightning_module = ConnectFourLightningModule(agent)
    
    trainer = torch.optim.Adam(lightning_module.parameters(), lr=1e-3)
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        logger.info("Starting self-play games...")
        data_module.generate_self_play_games()
        
        logger.info("Loading training data...")
        train_loader = DataLoader(data_module.dataset, batch_size=64, shuffle=True)
        
        logger.info("Starting training iteration...")
        for batch in train_loader:
            states, mcts_probs, rewards = batch
            outputs = lightning_module(states)
            loss = lightning_module.loss_function(outputs, mcts_probs, rewards)
            loss.backward()
            trainer.step()
            trainer.zero_grad()
        
        logger.info("Saving the model...")
        save_agent_model(agent)
        
    logger.info("Training completed.")

if __name__ == "__main__":
    train_alphazero(time_limit=0.1, num_self_play_games=2, load_model=False)