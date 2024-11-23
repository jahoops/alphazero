import logging
import os
import torch
import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..agent_code import AlphaZeroAgent

# Initialize logger
logger = logging.getLogger(__name__)

# Define model directory and path
MODEL_DIR = "nnbattle/agents/alphazero/model"
MODEL_PATH = os.path.join(MODEL_DIR, "alphazero_model_final.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_agent_model(agent: 'AlphaZeroAgent'):
    """
    Loads the agent's model from the predefined MODEL_PATH.

    :param agent: Instance of AlphaZeroAgent.
    """
    if not os.path.exists(MODEL_PATH):
        msg = f"Model path {MODEL_PATH} does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=agent.device)
        agent.model.load_state_dict(state_dict)
        agent.model_loaded = True
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        agent.model_loaded = False
        raise  # Re-raise the exception for test assertions

def save_agent_model(agent: 'AlphaZeroAgent', path: str = MODEL_PATH):
    """
    Saves the agent's model to the specified path.

    :param agent: Instance of AlphaZeroAgent.
    :param path: Destination path for the model weights.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = agent.model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved successfully to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise  # Re-raise the exception for test assertions

def preprocess_board(board_state: np.ndarray) -> torch.Tensor:
    """
    Converts the game board state into a format suitable for the neural network.
    
    :param board_state: numpy array representing the current board state
    :return: preprocessed board state as a torch tensor
    """
    # Convert to float32 and create a batch dimension
    processed = torch.FloatTensor(board_state).unsqueeze(0)
    return processed