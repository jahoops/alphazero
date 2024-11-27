def main():
    from nnbattle.utils.logger_config import set_log_level
    from nnbattle.agents.alphazero.train.train_alpha_zero import train_alphazero
    import logging
    import torch.multiprocessing as mp
    import os

    # Set up logging
    set_log_level(logging.INFO)

    # Set up CUDA and multiprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    # Run training
    try:
        train_alphazero(
            max_iterations=100,
            num_self_play_games=100,
            use_gpu=True,
            load_model=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        print("Training process has ended.")