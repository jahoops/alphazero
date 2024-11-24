# /agent_code.py

import logging

# Configure logging at the very start
logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more details
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
from nnbattle.constants import RED_TEAM, YEL_TEAM  # Ensure constants are imported
from .network import Connect4Net
from .utils.model_utils import load_agent_model, save_agent_model, MODEL_PATH
from .mcts import MCTSNode, mcts_simulate
from nnbattle.agents.base_agent import Agent  # Ensure this import is correct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class AlphaZeroAgent(Agent):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        action_dim,
        state_dim=2,
        use_gpu=False,
        num_simulations=800,
        c_puct=1.4,
        load_model=True,
        team=RED_TEAM  # Default to RED_TEAM
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_model_flag = load_model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.team = team
        self.memory = []
        self.model = Connect4Net(state_dim, action_dim).to(self.device)
        self.model_path = MODEL_PATH  # Added model_path attribute

        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        if load_model:
            try:
                load_agent_model(self)
            except FileNotFoundError:
                logger.warning("No model file found, starting with fresh model.")
                self.model_loaded = False

        logger.info("Initialized AlphaZeroAgent.")
        logger.info("AlphaZeroAgent initialized.")

    def log_gpu_stats(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9

            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
            logger.info(f"GPU Memory - Peak: {max_memory:.2f} GB")

            torch.cuda.reset_peak_memory_stats()

    def preprocess(self, board):
        board = board.copy()
        # Ensure board is in correct shape for processing
        if len(board.shape) == 3 and board.shape[0] > 2:
            board = board[:2]  # Take only first two channels if too many
        elif len(board.shape) == 2:
            # Create two channel board if single channel
            current_board = (board == self.team).astype(float)
            opponent_player = 2 if self.team == 1 else 1
            opponent_board = (board == opponent_player).astype(float)
            board = np.stack([current_board, opponent_board])
        
        # Ensure final shape is [2, 6, 7]
        assert board.shape == (2, 6, 7), f"Invalid board shape after preprocessing: {board.shape}"
        return torch.FloatTensor(board).to(self.device)

    def load_model_method(self):
        logger.warning("load_model method is deprecated. Use load_agent_model from utils.py instead.")
        load_agent_model(self)

    def save_model_method(self):
        save_agent_model(self, self.model_path)  # Pass the model_path explicitly

    def select_move(self, game: ConnectFourGame):
        logger.info(f"Agent {self.team} selecting a move.")

        # Ensure it's the agent's turn
        if game.last_team == self.team:
            raise InvalidTurnError(f"It's not Agent {self.team}'s turn.")

        # Get the current valid moves
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise InvalidMoveError("No valid moves available.")

        # Use MCTS or policy to select a move
        selected_action, action_probs = self.act(game, valid_moves)
        logger.debug(f"Agent {self.team} selected action {selected_action}.")

        return selected_action, action_probs

    def act(self, game: ConnectFourGame, valid_moves):
        selected_action, action_probs = mcts_simulate(self, game, valid_moves)
        if selected_action is None:
            raise InvalidMoveError("Failed to select a valid action.")
        return selected_action, action_probs

    def self_play(self, max_moves=100):
        game = ConnectFourGame()
        current_team = RED_TEAM  # Start with RED_TEAM

        # Initialize opponent_agent correctly
        opponent_agent = AlphaZeroAgent(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            use_gpu=self.device.type == 'cuda',
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            load_model=self.load_model_flag,
            team=YEL_TEAM if self.team == RED_TEAM else RED_TEAM
        )
        logger.info(f"Self-play initiated between Team {self.team} and Team {opponent_agent.team}")

        for move_number in range(max_moves):
            if game.get_game_state() != "ONGOING":
                logger.info(f"Game ended before reaching max moves: {game.get_game_state()}")
                break

            agent = self if self.team == current_team else opponent_agent
            logger.info(f"Move {move_number + 1}: Agent Team {agent.team} is making a move.")

            try:
                selected_action, _ = agent.select_move(game)
                logger.info(f"Agent Team {agent.team} selected action {selected_action}.")
                game.make_move(selected_action, agent.team)
                logger.info(f"Agent Team {agent.team} placed piece in column {selected_action}.")
            except (InvalidMoveError, InvalidTurnError) as e:
                logger.error(f"An error occurred during self-play: {e}")
                raise  # Stop the application

            # Alternate turns
            current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM
            logger.info(f"Next turn: Team {current_team}")
        
        # Handle the end of the game and update memory
        result = game.get_game_state()
        logger.info(f"Game Result: {result}")
        
        # Preprocess the final game state before appending
        preprocessed_state = self.preprocess(game.get_board())  # Shape: [2,6,7]
        mcts_prob = torch.zeros(self.action_dim, dtype=torch.float32)  # Initialize as Tensor
        self.memory.append((preprocessed_state, mcts_prob, result))

    def perform_training(self):
        """Perform training using train_alphazero.

        :param max_iterations: Maximum number of training iterations.
        """
        # Removed unnecessary import

        train_alphazero(
            max_iterations=1000,  # Replaced time_limit=3600 with max_iterations=1000
            num_self_play_games=1000,
            use_gpu=self.device.type == 'cuda',  # Ensure correct GPU usage
            load_model=self.load_model_flag
        )

    def initialize_agent_correctly(self):
        # Ensure the patch path is correct
        return initialize_agent(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            use_gpu=self.device.type == 'cuda',
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            load_model=self.load_model_flag
        )

def initialize_agent(
    action_dim=7,
    state_dim=2,
    use_gpu=False,
    num_simulations=800,
    c_puct=1.4,
    load_model=True
) -> AlphaZeroAgent:
    """
    Initializes and returns an instance of AlphaZeroAgent.

    :return: AlphaZeroAgent instance
    """
    return AlphaZeroAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        use_gpu=use_gpu,
        num_simulations=num_simulations,
        c_puct=c_puct,
        load_model=load_model
    )

__all__ = ['AlphaZeroAgent', 'initialize_agent']