def main():
    from nnbattle.agents.alphazero.agent_code import initialize_agent  # Added import
    from nnbattle.agents.alphazero.train import train_alphazero  # Updated import path
    from nnbattle.utils.logger_config import logger  # Corrected import
    import torch.multiprocessing as mp
    import os
    import torch
    import signal
    import sys

    # Initialize the agent
    agent = initialize_agent(
        action_dim=7,
        state_dim=3,
        use_gpu=True,
        num_simulations=800,
        c_puct=1.4,
        load_model=True
    )

    # Set up CUDA and multiprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal. Cleaning up...")
        try:
            # Clean up CUDA resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA memory cleaned up.")

            # Clean up multiprocessing resources
            mp.freeze_support()
            logger.info("Multiprocessing resources cleaned up.")

        finally:
            logger.info("Exiting program...")
            sys.exit(0)  # Force clean exit

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run training
    try:
        train_alphazero(
            agent=agent,  # Pass the agent as an argument
            max_iterations=10,
            num_self_play_games=10,
            use_gpu=True,
            load_model=True
        )
        # Save the trained model
        agent.save_model()  # Add this line
        logger.info("Training completed. Final model saved.")  # Add this line
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)  # Handle keyboard interrupt
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        signal_handler(signal.SIGTERM, None)  # Clean up on error

if __name__ == "__main__":
    main()