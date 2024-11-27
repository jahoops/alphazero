import logging
import os
import torch
import numpy as np
from typing import TYPE_CHECKING, Optional

# Use type hinting carefully to avoid circular imports
if TYPE_CHECKING:
    from ..agent_code import AlphaZeroAgent

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set appropriate logging level

# Define model directory and path
MODEL_DIR = "nnbattle/agents/alphazero/model"
MODEL_PATH = os.path.join(MODEL_DIR, "alphazero_model_final.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_agent_model(agent: 'AlphaZeroAgent'):
    """
    Loads the agent's model from the predefined MODEL_PATH.

    :param agent: Instance of AlphaZeroAgent.
    """
    try:
        state_dict = torch.load(agent.model_path, map_location=agent.device)
        agent.model.load_state_dict(state_dict)
        agent.model.to(agent.device)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.warning("Model file not found. Using initialized model.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Ensure the model remains initialized even if loading fails
    # Add check after attempting to load state_dict
    if agent.model is None:
        logger.error("Agent model is None after attempting to load the model.")

def save_agent_model(agent: 'AlphaZeroAgent', path: str = MODEL_PATH):
    """
    Saves the agent's model state dictionary to the specified path.

    :param agent: Instance of AlphaZeroAgent.
    :param path: Destination path for the model weights.
    """
    try:
        torch.save(agent.model.state_dict(), path)
        logger.info(f"Model saved successfully to {path}.")
    except Exception as e:
        logger.error(f"Failed to save the model to {path}: {e}")

def preprocess_board(board_state: np.ndarray, team: int) -> torch.Tensor:
    """
    Preprocesses the board state into a tensor suitable for the model.

    :param board_state: Numpy array representing the board.
    :param team: Current team's identifier.
    :return: Preprocessed tensor.
    """
    current_board = (board_state == team).astype(np.float32)
    opponent_board = (board_state == (3 - team)).astype(np.float32)
    valid_moves = np.zeros_like(current_board)
    for col in range(board_state.shape[1]):
        for row in range(board_state.shape[0]-1, -1, -1):
            if board_state[row][col] == EMPTY:
                valid_moves[row][col] = 1
                break
    state = np.stack([current_board, opponent_board, valid_moves])
    tensor = torch.from_numpy(state)
    return tensor.unsqueeze(0)  # Add batch dimension if required

__all__ = ['load_agent_model', 'save_agent_model', 'preprocess_board', 'MODEL_PATH']