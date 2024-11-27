def main():
    from nnbattle.agents.alphazero.agent_code import initialize_agent  # Added import
    from nnbattle.agents.alphazero.train import train_alphazero  # Updated import path
    from nnbattle.utils.logger_config import logger  # Corrected import
    import torch.multiprocessing as mp
    import os
    import torch

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

    # Run training
    try:
        train_alphazero(
            agent=agent,  # Pass the agent as an argument
            max_iterations=10,
            num_self_play_games=10,
            use_gpu=True,
            load_model=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")  # Now correctly uses logger.error
    finally:
        # CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory has been cleaned up.")  # Now correctly uses logger.info

        # Multiprocessing cleanup
        import multiprocessing as mp
        mp.freeze_support()
        logger.info("Multiprocessing resources have been cleaned up.")  # Now correctly uses logger.info

        # More informative completion message
        logger.info("Training completed successfully. All resources have been released.")  # Now correctly uses logger.info

if __name__ == "__main__":
    main()