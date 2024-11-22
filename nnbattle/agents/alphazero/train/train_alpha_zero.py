# nnbattle/agents/alphazero/train/train_alpha_zero.py

import os
import time
from datetime import timedelta
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from nnbattle.agents.alphazero import AlphaZeroAgent
from nnbattle.agents.alphazero.data_module import ConnectFourDataModule
from nnbattle.agents.alphazero.lightning_module import Connect4LightningModule
from nnbattle.game import ConnectFourGame
from nnbattle.agents.base_agent import Agent
from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent, initialize_agent  # Ensure AlphaZeroAgent is imported

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
        allocated = torch.cuda.memory_allocated()/1e9
        reserved = torch.cuda.memory_reserved()/1e9
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

def train_alphazero(time_limit, num_self_play_games=100, use_gpu=True, load_model=True):
    torch.set_float32_matmul_precision('medium')  # Set precision to utilize Tensor Cores
    start_time = time.time()
    
    # Initialize Agent with GPU support and control model loading
    state_dim = 2          # Updated to match number of channels
    action_dim = 7         # Seven possible actions
    num_simulations = 800  # Number of MCTS simulations
    agent = AlphaZeroAgent.initialize_agent()  # Use the imported initialize_agent
    
    # Log GPU usage during training
    if torch.cuda.is_available() and use_gpu:
        log_gpu_info(agent)
    
    # Perform Self-Play to populate agent's memory
    self_play(agent, num_self_play_games)
    
    # Initialize Data Module with the agent
    data_module = ConnectFourDataModule(agent=agent, batch_size=32)  # Adjust batch_size as needed
    
    # Initialize Lightning Module with state_dim and action_dim
    model = Connect4LightningModule(state_dim=state_dim, action_dim=action_dim)
    
    # Set up Model Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='alphazero/checkpoints/',
        filename='alphazero-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_time=timedelta(hours=time_limit),
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() and use_gpu else 'cpu',
        devices=1 if torch.cuda.is_available() and use_gpu else None,
    )
    
    # Start Training
    trainer.fit(model, datamodule=data_module)
    
    # Log GPU stats after training
    agent.log_gpu_stats()
    
    # Save the final model
    os.makedirs(os.path.dirname(agent.model_path), exist_ok=True)
    torch.save(model.state_dict(), agent.model_path)
    logger.info(f"Model saved to {agent.model_path}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed in {timedelta(seconds=elapsed_time)}")

if __name__ == "__main__":
    agent = AlphaZeroAgent.initialize_agent()
    train_alphazero(time_limit=0.1, num_self_play_games=2, load_model=False)