def main():
    from nnbattle.agents.alphazero.agent_code import initialize_agent  # Added import
    from nnbattle.agents.alphazero.train.train_alpha_zero import train_alphazero  # Updated import path
    from nnbattle.utils.logger_config import logger  # Corrected import
    from nnbattle.agents.alphazero.train import evaluate_agent  # Added import
    import torch
    import torch.multiprocessing as mp
    import os
    import torch
    import signal
    import sys
    import glob  # Added import

    # Hardware-optimized parameters
    initial_simulations = 50     # More initial sims since we have GPU power
    max_simulations = 400       # Higher max sims
    num_games_per_iteration = 50  # More parallel games
    test_games_per_eval = 20
    max_iterations = 5

    # Create model directory and backup untrained model
    model_dir = "nnbattle/agents/alphazero/model"
    os.makedirs(model_dir, exist_ok=True)
    untrained_model_path = os.path.join(model_dir, "untrained_baseline.pth")

    # Initialize agent with GPU optimization
    agent = initialize_agent(
        action_dim=7,
        state_dim=3,
        use_gpu=True,
        num_simulations=initial_simulations,
        c_puct=1.4
    )

    # Save untrained model for comparison
    torch.save(agent.model.state_dict(), untrained_model_path)
    logger.info(f"Saved untrained model to {untrained_model_path}")

    # Test untrained model against random
    logger.info("Testing untrained model against random player...")
    initial_performance = evaluate_agent(agent, num_games=test_games_per_eval, temperature=0.1)
    logger.info(f"Untrained model win rate: {initial_performance:.2f}")

    # Set up CUDA and multiprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal. Saving current state...")
        try:
            # Save the current model state
            interrupted_path = "nnbattle/agents/alphazero/model/interrupted_model.pth"
            torch.save(agent.model.state_dict(), interrupted_path)
            logger.info(f"Saved interrupted model to {interrupted_path}")

            # Clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleanup complete")
        finally:
            sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Add checkpoint directory
    checkpoint_dir = "nnbattle/agents/alphazero/model/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Run training with GPU optimized parameters
        train_alphazero(
            agent=agent,
            max_iterations=max_iterations,
            num_self_play_games=num_games_per_iteration,
            initial_simulations=initial_simulations,
            max_simulations=max_simulations,
            simulation_increase_interval=1,
            num_evaluation_games=test_games_per_eval,
            evaluation_frequency=1,
            use_gpu=True,
            save_checkpoint=True
        )
        
        # Test trained model
        logger.info("Testing trained model against random player...")
        final_performance = evaluate_agent(agent, num_games=test_games_per_eval)
        logger.info(f"Training Results:")
        logger.info(f"Initial win rate: {initial_performance:.2f}")
        logger.info(f"Final win rate: {final_performance:.2f}")
        logger.info(f"Improvement: {final_performance - initial_performance:.2f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        signal_handler(signal.SIGTERM, None)  # Clean up on error

if __name__ == "__main__":
    main()