def main():
    # Move all imports inside the function
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

    def train_alphazero(
        max_iterations: int = 1000,
        num_self_play_games: int = 1000,
        use_gpu: bool = False,
        load_model: bool = False,
        patience: int = 10
    ):
        """Trains the AlphaZero agent using self-play and reinforcement learning."""
        logger.info(f"Starting training with {max_iterations} iterations, {num_self_play_games} games per iteration")
        agent = initialize_agent(
            action_dim=7,
            state_dim=3,
            use_gpu=use_gpu,
            num_simulations=800,
            c_puct=1.4,
            load_model=load_model
        )

        if load_model:
            logger.info("Attempted to load existing model.")
        else:
            logger.info("Starting with a fresh model.")

        if agent.model is not None:
            logger.info("Agent model is successfully initialized.")
            agent.model.to(agent.device)
        else:
            logger.error("Agent model is None. Cannot proceed with training.")
            raise AttributeError("Agent model is not initialized.")

        log_gpu_info(agent)

        # Set up graceful shutdown handler
        import signal
        def signal_handler(signum, frame):
            logger.info("\nReceived shutdown signal. Cleaning up...")
            if 'trainer' in locals():
                trainer.should_stop = True
            torch.cuda.empty_cache()
            logger.info("Cleanup complete")

        signal.signal(signal.SIGINT, signal_handler)

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

    # Set logging level for all modules
    set_log_level(logging.INFO)  # Suppress INFO messages

    # Set the start method inside the main guard
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Ensure CUDA_VISIBLE_DEVICES is set
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Ensure multiprocessing is handled properly
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

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
        # Proper cleanup
        torch.cuda.empty_cache()
        # Clean up multiprocessing resources
        for p in mp.active_children():
            p.terminate()
        logger.info("Resources released and cleanup complete.")

if __name__ == "__main__":
    main()