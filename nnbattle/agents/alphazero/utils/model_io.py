
import logging
import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import AlphaZeroAgent  # Adjusted import to prevent circular dependency

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
                map_location=agent.device
            )
        )
        logging.getLogger(__name__).info(f"Model loaded successfully from {MODEL_PATH}")
        agent.model_loaded = True
    else:
        logging.getLogger(__name__).error(f"Model path {MODEL_PATH} does not exist.")
        raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist.")

def save_agent_model(agent: "AlphaZeroAgent", path: str = MODEL_PATH):
    """
    Saves the agent's model to the specified path.

    :param agent: Instance of AlphaZeroAgent.
    :param path: Destination path for the model weights.
    """
    torch.save(agent.model.state_dict(), path)
    logging.getLogger(__name__).info(f"Model saved to {path}.")