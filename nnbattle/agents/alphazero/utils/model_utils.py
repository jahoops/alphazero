import logging
import os
import torch
import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..agent_code import AlphaZeroAgent

# Create model directory if it doesn't exist
MODEL_DIR = os.path.join("nnbattle", "agents", "alphazero", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "alphazero_model_final.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_agent_model(agent: 'AlphaZeroAgent'):
    """
    Loads the agent's model from the predefined MODEL_PATH.

    :param agent: Instance of AlphaZeroAgent.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")
    # Address the FutureWarning by setting weights_only=True
    agent.model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=agent.device,
            weights_only=True  # Added parameter to suppress warning
        )
    )
    agent.model_loaded = True

def save_agent_model(agent: 'AlphaZeroAgent', path: str = MODEL_PATH):
    """
    Saves the agent's model to the specified path.

    :param agent: Instance of AlphaZeroAgent.
    :param path: Destination path for the model weights.
    """
    torch.save(agent.model.state_dict(), path)
    logging.getLogger(__name__).info(f"Model saved to {path}.")

def preprocess_board(board_state: np.ndarray) -> torch.Tensor:
    """
    Converts the game board state into a format suitable for the neural network.
    
    :param board_state: numpy array representing the current board state
    :return: preprocessed board state as a torch tensor
    """
    # Convert to float32 and create a batch dimension
    processed = torch.FloatTensor(board_state).unsqueeze(0)
    return processed