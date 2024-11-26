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
from nnbattle.agents.base_agent import BaseAgent  # Ensure this import is correct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class AlphaZeroAgent(BaseAgent):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        action_dim,
        state_dim=3,  # Changed from 2 to 3 to match the state tensor shape
        use_gpu=True,
        num_simulations=800,
        c_puct=1.4,
        load_model=True,
        team=RED_TEAM  # Ensure team is initialized
    ):
        super().__init__(team)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_loaded = True
        self.load_model_flag = load_model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.team = team  # Initialize as an instance attribute
        self.memory = []
        self.model = Connect4Net(state_dim, action_dim)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.data = param.data.to(self.device)
        
        self.model_path = MODEL_PATH  # Added model_path attribute

        logger.info(f"Initialized AlphaZeroAgent on device: {self.device}")
        
        if load_model:
            try:
                load_agent_model(self)
                self.model_loaded = True
                logger.info("Model loaded successfully.")
            except FileNotFoundError:
                logger.warning("No model file found, starting with a fresh model.")
                self.model_loaded = False

        logger.info("Initialized AlphaZeroAgent.")

    def log_gpu_stats(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9

            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
            logger.info(f"GPU Memory - Peak: {max_memory:.2f} GB")

            torch.cuda.reset_peak_memory_stats()

    def preprocess(self, board, to_device=None):
        board = board.copy()
        # Create a 3-channel state representation
        # Channel 1: Current player's pieces
        # Channel 2: Opponent's pieces
        # Channel 3: Valid moves (binary mask)
        current_board = (board == self.team).astype(np.float32)
        opponent_board = (board == (3 - self.team)).astype(np.float32)
        valid_moves = np.zeros_like(current_board)
        for col in range(board.shape[1]):
            for row in range(board.shape[0]-1, -1, -1):
                if board[row][col] == 0:
                    valid_moves[row][col] = 1
                    break
        
        state = np.stack([current_board, opponent_board, valid_moves])
        tensor = torch.FloatTensor(state)
        
        if to_device:
            return tensor.to(to_device)
        return tensor.cpu()

    def select_move(self, game: ConnectFourGame, temperature=1.0):
        logger.info(f"Agent {self.team} selecting a move.")

        # Ensure it's the agent's turn
        if game.last_team is not None:  # Only check if not first move
            if game.last_team == self.team:
                logger.error(f"It's not Agent {self.team}'s turn.")
                raise InvalidTurnError(f"It's not Agent {self.team}'s turn.")

        # Get the current valid moves
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            logger.error("No valid moves available.")
            raise InvalidMoveError("No valid moves available.")

        # Use MCTS or policy to select a move
        selected_action, action_probs = self.act(game, valid_moves, temperature=temperature)
        logger.info(f"Agent {self.team} selected action {selected_action}.")

        return selected_action, action_probs

    def act(self, game: ConnectFourGame, valid_moves, temperature=1.0, **kwargs):
        self.model.eval()  # Set to evaluation mode
        self.model = self.model.to(self.device)  # Ensure model is on correct device
        logger.debug(f"Model device: {next(self.model.parameters()).device}")
        logger.debug("Starting MCTS simulation for action selection.")
        selected_action, action_probs = mcts_simulate(self, game, valid_moves, temperature=temperature)
        if selected_action is None:
            logger.error("Failed to select a valid action.")
            raise InvalidMoveError("Failed to select a valid action.")
        logger.debug(f"MCTS simulation completed. Selected Action: {selected_action}")
        return selected_action, action_probs

    def self_play(self, max_moves=100, temperature=1.0):
        game = ConnectFourGame()
        current_team = RED_TEAM
        original_team = self.team
        game_history = []  # Store (state, mcts_prob, current_team) tuples

        try:
            while game.get_game_state() == "ONGOING" and len(game_history) < max_moves:
                self.team = current_team
                state = self.preprocess(game.get_board())
                
                # Use higher temperature early in the game for exploration
                current_temp = 1.0 if len(game_history) < 10 else temperature
                selected_action, action_prob = self.select_move(game, temperature=current_temp)
                
                # Store state, action probability, and current team
                game_history.append((state.cpu(), action_prob.cpu(), current_team))
                
                game.make_move(selected_action, current_team)
                current_team = 3 - current_team
        finally:
            self.team = original_team

        # Get final result and assign rewards
        result = game.get_game_state()
        
        # Calculate rewards with a discount factor for move timing
        discount = 0.99
        for idx, (state, mcts_prob, team) in enumerate(reversed(game_history)):
            if result == "Draw":
                reward = 0.0
            else:
                # Winning moves get higher rewards if they end the game sooner
                reward = discount ** idx if result == team else -(discount ** idx)
            
            self.memory.append((state, mcts_prob, float(reward)))

        logger.info(f"Game completed with {len(game_history)} moves, result: {result}")
        return len(game_history)

    def perform_training(self):
        """Perform training using train_alphazero.

        :param max_iterations: Maximum number of training iterations.
        """
        logger.info("Commencing training procedure.")
        train_alphazero(
            max_iterations=100,          # Set to a reasonable number
            num_self_play_games=100,     # Set to a reasonable number
            use_gpu=self.device.type == 'cuda',  # Ensure correct GPU usage
            load_model=self.load_model_flag
        )
        logger.info("Training procedure completed.")

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
                action, _ = self.select_move(game)
                game.make_move(action, self.team)
                if game.get_game_state() != "ONGOING":
                    break
                opponent_action = np.random.choice(game.get_valid_moves())
                game.make_move(opponent_action, 3 - self.team)
            result = game.get_game_state()
            if result == self.team:
                wins += 1
            elif result == "Draw":
                draws += 1
            else:
                losses += 1
        logger.info(f"Evaluation Results over {num_evaluations} games: Wins={wins}, Draws={draws}, Losses={losses}")

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
    agent = AlphaZeroAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        use_gpu=use_gpu,
        num_simulations=num_simulations,
        c_puct=c_puct,
        load_model=load_model
    )
    return agent

__all__ = ['AlphaZeroAgent', 'initialize_agent']