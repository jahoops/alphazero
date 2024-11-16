# alphazero_agent/train/train_alpha_zero.py

import logging
import time
from alphazero_agent import AlphaZeroAgent
from connect_four_game import ConnectFourGame  # Update the import path as needed
import os

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def initialize_agent(state_dim, action_dim, use_gpu=False, model_path="alphazero_agent/model/alphazero_model_final.pth"):
    """
    Initialize the AlphaZeroAgent.

    :param state_dim: Dimension of the game state.
    :param action_dim: Number of possible actions.
    :param use_gpu: Whether to use GPU for training.
    :param model_path: Path to the trained model.
    :return: Initialized AlphaZeroAgent instance.
    """
    return AlphaZeroAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_gpu=use_gpu,
        model_path=model_path
    )

def load_agent_model(agent, model_path):
    """
    Load the model weights into the agent.

    :param agent: Instance of AlphaZeroAgent.
    :param model_path: Path to the model file.
    """
    try:
        agent.load_model()
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

def save_agent_model(agent, model_path):
    """
    Save the agent's model weights.

    :param agent: Instance of AlphaZeroAgent.
    :param model_path: Path to save the model file.
    """
    try:
        agent.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

def perform_training(agent, time_limit):
    """
    Perform the training loop for the agent.

    :param agent: Instance of AlphaZeroAgent.
    :param time_limit: Time limit for training in seconds.
    """
    start_time = time.time()
    while time.time() - start_time < time_limit:
        try:
            # Perform self-play to generate training data
            agent.self_play()

            # Perform a training step
            agent.train_step(batch_size=32)

            elapsed = time.time() - start_time
            logger.info(f"Elapsed time: {elapsed:.2f}s | Memory Size: {len(agent.memory)}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during training: {e}")

def train_alphazero(time_limit, load_model=True, model_path="alphazero_agent/model/alphazero_model_final.pth"):
    """
    Train the AlphaZeroAgent.

    :param time_limit: Total time for training in seconds.
    :param load_model: Whether to load an existing model.
    :param model_path: Path to load/save the model.
    """
    state_dim = 42  # Example: 6 rows * 7 columns
    action_dim = 7  # Number of columns in Connect Four
    agent = initialize_agent(state_dim=state_dim, action_dim=action_dim, use_gpu=False, model_path=model_path)

    if load_model:
        load_agent_model(agent, model_path)

    perform_training(agent, time_limit)

    save_agent_model(agent, model_path)

if __name__ == "__main__":
    # Example: Train for 1 hour (3600 seconds)
    train_alphazero(time_limit=3600, load_model=True)