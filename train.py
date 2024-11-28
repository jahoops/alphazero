def main():
    from nnbattle.agents.alphazero.agent_code import initialize_agent  # Added import
    from nnbattle.agents.alphazero.train import train_alphazero  # Updated import path
    from nnbattle.utils.logger_config import logger  # Corrected import
    import torch
    import torch.multiprocessing as mp
    import os
    import torch
    import signal
    import sys
    import glob  # Added import

    # Progressive MCTS simulations based on training stage
    initial_simulations = 10  # Start with very few simulations
    max_simulations = 800    # Maximum simulations for later training

    # Initialize the agent with minimal simulations
    agent = initialize_agent(
        action_dim=7,
        state_dim=3,
        use_gpu=True,
        num_simulations=initial_simulations,  # Start with few simulations
        c_puct=1.4,
        load_model=True
    )

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

    # Add checkpoint handling
    checkpoint_dir = "nnbattle/agents/alphazero/model/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Look for latest checkpoint
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint_*.pth"))
    start_iteration = 0
    if checkpoints and load_model:
        latest_checkpoint = checkpoints[-1]
        iteration_num = int(latest_checkpoint.split('_')[-1].split('.')[0])
        logger.info(f"Found checkpoint from iteration {iteration_num}")
        agent.model_path = latest_checkpoint
        start_iteration = iteration_num + 1

    # Run training with balanced parameters
    try:
        os.makedirs("nnbattle/agents/alphazero/model", exist_ok=True)
        
        # Run training with smaller batches but more frequent evaluation
        train_alphazero(
            agent=agent,
            max_iterations=10,
            num_self_play_games=50,
            initial_simulations=initial_simulations,
            max_simulations=max_simulations,
            simulation_increase_interval=2,  # Increase simulations every 2 iterations
            use_gpu=True,
            load_model=True,
            save_checkpoint=True,
            checkpoint_frequency=1
        )
        
        # Save final model
        final_model_path = "nnbattle/agents/alphazero/model/alphazero_model_final.pth"
        torch.save(agent.model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        signal_handler(signal.SIGTERM, None)  # Clean up on error

if __name__ == "__main__":
    main()