def evaluate_agent(agent, num_games=20):
    """Evaluate agent performance against random opponent."""
    wins = 0
    draws = 0
    total_games = 0
    
    for _ in range(num_games):
        game = ConnectFourGame()
        while game.get_game_state() == "ONGOING":
            try:
                if game.last_team == agent.team:
                    action, _ = agent.select_move(game, agent.team, temperature=0.1)
                else:
                    valid_moves = game.get_valid_moves()
                    action = np.random.choice(valid_moves)
                game.make_move(action, game.last_team if game.last_team else RED_TEAM)
            except Exception as e:
                logger.error(f"Error during evaluation game: {e}")
                break
        
        result = game.get_game_state()
        if result == agent.team:
            wins += 1
        elif result == "Draw":
            draws += 1
        total_games += 1
    
    return wins / total_games if total_games > 0 else 0.0

def train_alphazero(
    agent,  # Added agent parameter
    max_iterations: int = 1000,
    num_self_play_games: int = 1000,
    use_gpu: bool = False,
    load_model: bool = False,
    patience: int = 10
):
    """Trains the AlphaZero agent using self-play and reinforcement learning."""
    from ....utils.logger_config import logger, set_log_level
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
    from ..agent_code import initialize_agent  # Moved import here
    from nnbattle.constants import RED_TEAM, YEL_TEAM
    from ..data_module import ConnectFourDataModule, ConnectFourDataset  # Moved import here
    from ..lightning_module import ConnectFourLightningModule
    from ..utils.model_utils import (
        MODEL_PATH,
        load_agent_model,
        save_agent_model
    )
    from nnbattle.agents.alphazero.self_play import SelfPlay  # Import from new location if needed

    # Remove any existing logging configuration
    import signal
    def signal_handler(signum, frame):
        logger.info("\nReceived shutdown signal. Cleaning up...")
        if 'trainer' in locals():
            trainer.should_stop = True
        torch.cuda.empty_cache()
        logger.info("Cleanup complete")

    signal.signal(signal.SIGINT, signal_handler)
    
    def log_gpu_info(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9

            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
            logger.info(f"GPU Memory - Peak: {max_memory:.2f} GB")

            torch.cuda.reset_peak_memory_stats()

    # Modify DataLoader settings to be more stable
    data_module = ConnectFourDataModule(
        agent, 
        num_games=num_self_play_games,
        batch_size=32,
        num_workers=2,  # Reduce from 4 to 2 for stability
        persistent_workers=True
    )

    lightning_module = ConnectFourLightningModule(agent)  # Create the lightning module instance

    # Generate self-play games using the SelfPlay class
    game = ConnectFourGame()
    self_play = SelfPlay(game=game, model=agent.model, num_simulations=agent.num_simulations)
    training_data = self_play.generate_training_data(num_self_play_games)

    # Check if we got valid training data
    if not training_data:
        logger.error("No valid training data generated")
        raise ValueError("No valid training data generated")

    data_module.dataset = ConnectFourDataset(training_data, agent)

    logger.info(f"Generated {len(data_module.dataset)} training examples.")

    data_module.setup('fit')

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

    # Create trainer with proper cleanup settings
    trainer = pl.Trainer(
        max_epochs=10,  # Increased from 1 to 10 to allow more training on each batch
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger_tb,
        log_every_n_steps=10,
        detect_anomaly=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=2,
        reload_dataloaders_every_n_epochs=1  # Reload to prevent worker issues
    )

    # Disable model loading in agent's select_move during training
    agent.load_model_flag = False

    best_performance = 0.0
    no_improvement_count = 0

    try:
        # Set proper CUDA tensor sharing strategy
        if use_gpu and torch.cuda.is_available():
            torch.multiprocessing.set_sharing_strategy('file_system')
        
        for iteration in range(1, max_iterations + 1):
            performance = None  # Initialize performance at start of each iteration
            # Adjust temperature parameter
            if iteration < max_iterations * 0.5:
                temperature = 1.0  # High temperature for more exploration
            else:
                temperature = 0.1  # Low temperature for more exploitation

            # Adjust c_puct parameter
            agent.c_puct = 1.4 if iteration < max_iterations * 0.5 else 1.0

            # Alternate starting team each iteration
            starting_team = RED_TEAM if iteration % 2 == 1 else YEL_TEAM

            logger.info(f"=== Starting Training Iteration {iteration}/{max_iterations} ===")
            logger.info(f"Using temperature: {temperature}, c_puct: {agent.c_puct}")
            logger.info(f"Starting team: {starting_team}")
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
                    # Ensure CUDA tensors are properly handled
                    with torch.cuda.device(agent.device):
                        performance = evaluate_agent(agent, num_games=20)
                        torch.cuda.empty_cache()  # Clear cache after evaluation
                    
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
                # Clear CUDA cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            finally:
                logger.info(f"Iteration {iteration} completed in {time.time() - iteration_start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Ensure proper cleanup of CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("=== Training Completed Successfully ===")

# Ensure __all__ is defined
__all__ = ['train_alphazero']