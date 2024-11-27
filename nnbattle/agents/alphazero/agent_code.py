# /agent_code.py

import logging

# Configure logging at the very start
logging.basicConfig(
    level=logging.DEBUG,  # Or DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Proceed with your imports
import os
import time
from datetime import timedelta
import torch
import pytorch_lightning as pl
import numpy as np
import copy
from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from nnbattle.constants import RED_TEAM, YEL_TEAM, EMPTY  # Add EMPTY to imports
from .network import Connect4Net
from .utils.model_utils import load_agent_model, save_agent_model, MODEL_PATH
from .mcts import MCTSNode, mcts_simulate
from ..base_agent import BaseAgent  # Adjusted import
from nnbattle.utils.logger_config import logger

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

from contextlib import contextmanager

@contextmanager
def model_mode(model, training):
    original_mode = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(original_mode)

class AlphaZeroAgent(BaseAgent):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        action_dim,
        state_dim=3,  # Adjusted state dimension
        use_gpu=True,
        num_simulations=800,
        c_puct=1.4,
        load_model=True,
        team=RED_TEAM  # Initialized team
    ):
        super().__init__(team)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.load_model_flag = load_model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.team = team
        self.memory = []
        self.model_path = MODEL_PATH  # Added model_path attribute

        # Initialize the model
        self.model = Connect4Net(state_dim, action_dim).to(self.device)
        logger.info("Model instance created.")

        if load_model:
            try:
                load_agent_model(self)
            except FileNotFoundError:
                logger.warning("No model file found, using the initialized model.")
            except Exception as e:
                logger.error(f"An error occurred while loading the model: {e}")
                # Ensure the model remains initialized even if loading fails

        # Log initialization after attempting to load the model
        logger.info(f"Initialized AlphaZeroAgent on device: {self.device}")
        logger.info("AlphaZeroAgent setup complete.")

        # After model initialization and model loading
        logger.debug(f"After initialization, self.model is {self.model}")
        logger.debug(f"Model type: {type(self.model)}")

        # Check if self.model is None
        if self.model is None:
            logger.error("Agent model is None after initialization.")
        else:
            logger.info("Agent model is properly initialized.")

    def save_model(self):
        save_agent_model(self)

    def log_gpu_stats(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9

            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
            logger.info(f"GPU Memory - Peak: {max_memory:.2f} GB")

            torch.cuda.reset_peak_memory_stats()

    def preprocess(self, board, team, to_device=None):
        board = board.copy()
        # Create a 3-channel state representation
        current_board = (board == team).astype(np.float32)
        opponent_board = (board == (3 - team)).astype(np.float32)
        valid_moves = np.zeros_like(current_board)
        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 1, -1, -1):
                if board[row][col] == EMPTY:  # Now EMPTY is defined
                    valid_moves[row][col] = 1
                    break
        
        state = np.stack([current_board, opponent_board, valid_moves])
        tensor = torch.FloatTensor(state)

        if to_device:
            return tensor.to(to_device)
        return tensor.cpu()

    def select_move(self, game: ConnectFourGame, team: int, temperature=1.0):
        #logger.info(f"Agent {team} selecting a move.")
        # Ensure it's the correct team's turn
        if game.last_team is not None:  # Only check if not first move
            if game.last_team == team:
                logger.error(f"It's not Team {team}'s turn.")
                raise InvalidTurnError(f"It's not Team {team}'s turn.")

        # Get the current valid moves
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            logger.error("No valid moves available.")
            raise InvalidMoveError("No valid moves available.")

        if self.model is None:
            logger.error("Agent model is not initialized.")
            raise AttributeError("Agent model is not initialized.")

        selected_action, action_probs = self.act(game, valid_moves, team, temperature=temperature)
        return selected_action, action_probs

    def evaluate_model(self):
        """Evaluate the model on a set of validation games to monitor learning progress."""
        logger.info("Starting model evaluation.")
        # Implement evaluation logic, e.g., play against a random agent
        wins = 0
        draws = 0
        losses = 0
        num_evaluations = 100
        for _ in range(num_evaluations):
            game = ConnectFourGame()
            while game.get_game_state() == "ONGOING":
                if game.last_team == self.team:
                    action, _ = self.select_move(game, temperature=0)
                else:
                    valid_moves = game.get_valid_moves()
                    action = np.random.choice(valid_moves)
                game.make_move(action, game.last_team if game.last_team else RED_TEAM)
            result = game.get_game_state()
            if result == self.team:
                wins += 1
            elif result == "Draw":
                draws += 1
            else:
                losses += 1
        logger.info(f"Evaluation Results over {num_evaluations} games: Wins={wins}, Draws={draws}, Losses={losses}")

    def act(self, game: ConnectFourGame, valid_moves, team: int, temperature=1.0, **kwargs):
        with model_mode(self.model, training=False):  # Temporarily set to eval mode
            selected_action, action_probs = mcts_simulate(self, game, valid_moves, team, temperature=temperature)
        return selected_action, action_probs

def initialize_agent(
    action_dim=7,
    state_dim=3,  # Ensure this matches
    use_gpu=True,
    num_simulations=800,
    c_puct=1.4,
    load_model=True
) -> AlphaZeroAgent:
    agent = AlphaZeroAgent(        action_dim=action_dim,
        state_dim=state_dim,
        use_gpu=use_gpu,        
        num_simulations=num_simulations,        
        c_puct=c_puct,        
        load_model=load_model
    )
    return agent

__all__ = ['AlphaZeroAgent', 'initialize_agent']