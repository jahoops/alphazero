
from nnbattle.utils.logger_config import logger, set_log_level
import logging

# Set global log level at the start of your program
set_log_level(logging.INFO)

import os
import time
from datetime import timedelta
import torch
import numpy as np
import pytorch_lightning as pl
from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from nnbattle.agents.alphazero.agent_code import initialize_agent  # Moved import here
from nnbattle.constants import RED_TEAM
from nnbattle.agents.alphazero.data_module import ConnectFourDataModule
from nnbattle.agents.alphazero.lightning_module import ConnectFourLightningModule
from nnbattle.agents.alphazero.utils.model_utils import (
    MODEL_PATH,
    load_agent_model,
    save_agent_model
)

# Remove any existing logging configuration
# ...existing code...

# Set float32 matmul precision for Tensor Cores
torch.set_float32_matmul_precision('high')

def log_gpu_info(agent):
    """Log GPU information."""
    if torch.cuda.is_available():
        logger.warning(f"Training on GPU: {torch.cuda.get_device_name()}")
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.warning(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        logger.warning("No CUDA device available")

def self_play(agent, num_games):
    memory = []
    game = ConnectFourGame()
    game_history = []
    for game_num in range(num_games):
        game.reset()
        agent.team = RED_TEAM  # Initialize current player at the start of the game
        logger.info(f"Starting game {game_num + 1}/{num_games}")
        game_start_time = time.time()
        while game.get_game_state() == "ONGOING":
            # Pass the temperature parameter
            selected_action, action_probs = agent.select_move(game, temperature=1.0)
            logger.info(f"Move {selected_action}/{agent.team}")
            game_history.append((game.get_board(), action_probs, agent.team))
            agent.team = 3 - agent.team  # switch team
        game_end_time = time.time()
        logger.info(f"Time taken for game {game_num + 1}: {game_end_time - game_start_time:.4f} seconds")
        result = game.get_game_state()
        for idx, (state, mcts_prob, team) in enumerate(reversed(game_history)):
            if result == "Draw":
                reward = 0.0
            else:
                reward = 1.0 if result == team else -1.0  # Corrected reward assignment
            memory.append((state, mcts_prob, float(reward)))
        # Ensure that agent.memory has been populated
        if not agent.memory:
            logger.warning("No self-play games were generated.")
        else:
            # Ensure that state is preprocessed correctly
            preprocessed_state = agent.preprocess(game.get_board())  # Changed from game.get_state() to game.get_board()
            mcts_prob = torch.zeros(agent.action_dim, dtype=torch.float32)  # Initialize as Tensor
            memory.append((preprocessed_state, mcts_prob, reward))
            logger.info(f"Finished game {game_num + 1}/{num_games} with result: {result}")
    agent.memory.extend(memory)  # Assuming agent.memory is a list
    return memory

def train_alphazero(
    max_iterations: int = 1000,
    num_self_play_games: int = 1000,
    use_gpu: bool = False,
    load_model: bool = False,
    patience: int = 10
):
    """Trains the AlphaZero agent using self-play and reinforcement learning."""
    logger.info(f"Starting training with {max_iterations} iterations, {num_self_play_games} games per iteration")
    # Only load the model once at the start if requested
    agent = initialize_agent(
        action_dim=7,
        state_dim=3,  # Changed from 2 to 3
        use_gpu=use_gpu,
        num_simulations=800,
        c_puct=1.4,
        load_model=load_model  # Only load once at initialization
    )
       
    # Add logging based on whether the model was loaded
    if agent.model_loaded:
        logger.info("Loaded existing model successfully.")
    else:
        logger.info("No existing model found. Starting with a fresh model.")
    
    # Remove the following error raising
    # if agent.model is None or not agent.model_loaded:
    #     logger.error("Agent model is not initialized.")
    #     raise AttributeError("Agent model is not initialized.")
    
    if agent.model is not None:
        agent.model.to(agent.device)  # Ensure the model is on the correct device
    else:
        logger.error("Agent model is None. Cannot proceed with training.")
        raise AttributeError("Agent model is not initialized.")
    
    log_gpu_info(agent)  # Log initial GPU state
    
    data_module = ConnectFourDataModule(agent, num_self_play_games)
    lightning_module = ConnectFourLightningModule(agent)
    
    # Generate self-play games **before** setting up the data
    logger.info("Generating initial self-play games...")
    data_module.generate_self_play_games(temperature=1.0)
    
    # Check if dataset has data before setting up
    if len(data_module.dataset) == 0:
        logger.error("No self-play games were generated. Training cannot proceed.")
        raise ValueError("Self-play generation failed. No data available for training.")
    
    data_module.setup('fit')  # Explicitly set up the data after generation
    
    # Add checkpoint callback to save best models
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='nnbattle/agents/alphazero/model/checkpoints',
        filename='alphazero-{epoch:02d}-{loss:.2f}',  # Updated to log 'loss'
        save_top_k=3,
        monitor='loss',  # Changed from 'train_loss' to 'loss'
        mode='min'
    )
    
    # Add Early Stopping callback (Optional)
    from pytorch_lightning.callbacks import EarlyStopping

    # Update the EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor='loss',     # Change from 'train_loss' to 'loss'
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    
    from pytorch_lightning.loggers import TensorBoardLogger  # Add TensorBoard logger import

    # Initialize TensorBoard logger with more details
    logger_tb = TensorBoardLogger(
        save_dir="tb_logs",
        name="alphazero",
        version=None,  # Auto-increment version
        default_hp_metric=False  # Don't log hp_metric
    )

    trainer = pl.Trainer(
        max_epochs=10,  # Increased from 1 to 10 to allow more training on each batch
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger_tb,
        log_every_n_steps=10,
        detect_anomaly=False
    )
    
    # Disable model loading in agent's select_move during training
    agent.load_model_flag = False
    
    best_performance = 0.0
    no_improvement_count = 0
    
    for iteration in range(1, max_iterations + 1):
        performance = None  # Initialize performance at start of each iteration
        # Adjust temperature parameter
        if iteration < max_iterations * 0.5:
            temperature = 1.0  # High temperature for more exploration
        else:
            temperature = 0.1  # Low temperature for more exploitation

        # Adjust c_puct parameter
        agent.c_puct = 1.4 if iteration < max_iterations * 0.5 else 1.0

        logger.info(f"=== Starting Training Iteration {iteration}/{max_iterations} ===")
        logger.info(f"Using temperature: {temperature}, c_puct: {agent.c_puct}")
        iteration_start_time = time.time()
        try:
            logger.info(f"Generating {num_self_play_games} self-play games...")
            log_gpu_info(agent)  # Before generating self-play games
            data_module.generate_self_play_games(temperature=temperature)
            
            if len(data_module.dataset) == 0:
                # Handle the case where no data was generated
                logger.error("No data generated during self-play. Skipping this iteration.")
                continue
            
            # Proceed with training
            trainer.fit(lightning_module, datamodule=data_module)
            
            # Optionally evaluate the model
            if iteration % 10 == 0:
                performance = evaluate_agent(agent, num_games=20)
                logger.info(f"Iteration {iteration}: Evaluation Performance: {performance}")
                if performance > best_performance:
                    best_performance = performance
                    no_improvement_count = 0
                    # Optionally save the best model
                    save_agent_model(agent)
                    logger.info(f"New best performance: {best_performance}. Model saved.")
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement in performance. ({no_improvement_count}/{patience})")
                    if no_improvement_count >= patience:
                        logger.info("Early stopping triggered due to no improvement.")
                        break

        except (InvalidMoveError, InvalidTurnError) as e:
            logger.error(f"Game error during iteration {iteration}: {e}")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred during iteration {iteration}: {e}")
            continue
        finally:
            logger.info(f"Iteration {iteration} completed in {time.time() - iteration_start_time:.2f} seconds")

    logger.info("=== Training Completed Successfully ===")

def evaluate_agent(agent, num_games=20):
    """Evaluate the agent's performance against a random opponent."""
    wins = 0
    for _ in range(num_games):
        result = play_game(agent)
        if result == agent.team:
            wins += 1
    performance = wins / num_games
    return performance

def play_game(agent):
    """Play a single game against a random opponent."""
    game = ConnectFourGame()
    current_team = RED_TEAM  # Start with RED_TEAM
    while game.get_game_state() == "ONGOING":
        if current_team == agent.team:
            action, _ = agent.select_move(game, temperature=0)
        else:
            valid_moves = game.get_valid_moves()
            action = np.random.choice(valid_moves)
        game.make_move(action, current_team)
        current_team = 3 - current_team  # Switch teams
    return game.get_game_state()

if __name__ == "__main__":
    # Set logging level for all modules
    set_log_level(logging.INFO)  # Suppress INFO messages
    
    # Set the start method inside the main guard
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Ensure CUDA_VISIBLE_DEVICES is set
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        train_alphazero(
            max_iterations=100,
            num_self_play_games=100,
            use_gpu=True,
            load_model=True
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")    
    finally:
        # Release all CUDA resources
        torch.cuda.empty_cache()
        logger.info("CUDA resources have been released.")