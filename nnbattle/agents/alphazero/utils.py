# /utils.py

from __future__ import annotations  # Enable postponed evaluation of annotations
import copy
import logging
import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_code import AlphaZeroAgent  # Prevent circular import at runtime

def preprocess_board(board, player):
    """
    Preprocesses the board by applying the player's perspective.

    :param board: Current game board as a NumPy array.
    :param player: Current player (1 or -1).
    :return: Preprocessed board as a Torch tensor.
    """
    return copy.deepcopy(board) * player

def deepcopy_env(env):
    """
    Creates a deep copy of the game environment.

    :param env: Current game environment.
    :return: A deep copy of the game environment.
    """
    return copy.deepcopy(env)

# New: Define model_path centrally
MODEL_PATH = "nnbattle/agents/alphazero/model/alphazero_model_final.pth"

def load_agent_model(agent: "AlphaZeroAgent"):
    """
    Loads the agent's model from the predefined MODEL_PATH.

    :param agent: Instance of AlphaZeroAgent.
    """
    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=agent.device,
                weights_only=True  # Add this parameter if applicable
            )
        )
        logging.getLogger(__name__).info(f"Model loaded successfully from {MODEL_PATH}")
        agent.model_loaded = True
    else:
        logging.getLogger(__name__).error(f"Model path {MODEL_PATH} does not exist.")
        raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist.")

def save_agent_model(agent: AlphaZeroAgent, path: str = MODEL_PATH):
    """
    Saves the agent's model to the specified path.

    :param agent: Instance of AlphaZeroAgent.
    :param path: Destination path for the model weights.
    """
    torch.save(agent.model.state_dict(), path)
    logging.getLogger(__name__).info(f"Model saved to {path}.")

def initialize_agent(
    action_dim=7,
    state_dim=2,
    use_gpu=False,
    # model_path="nnbattle/agents/alphazero/model/alphazero_model_final.pth",  # Removed parameter
    num_simulations=800,
    c_puct=1.4,
    load_model=True
) -> "AlphaZeroAgent":
    """
    Initializes and returns an instance of AlphaZeroAgent.

    :return: AlphaZeroAgent instance
    """
    from .agent_code import AlphaZeroAgent  # Ensure correct import
    return AlphaZeroAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        use_gpu=use_gpu,
        # model_path=model_path,  # Removed to use centralized MODEL_PATH
        num_simulations=num_simulations,
        c_puct=c_puct,
        load_model=load_model
    )

# ...existing code...