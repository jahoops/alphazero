# /agent_code.py

import logging
from collections import deque
import os
import time
from unittest.mock import MagicMock  # Add this import
from nnbattle.agents.alphazero.mcts import MCTSNode  # Add this import

import numpy as np
import torch

from nnbattle.game.connect_four_game import ConnectFourGame  # Update the import path as needed

from .network import Connect4Net
from nnbattle.agents.base_agent import Agent
from .utils import deepcopy_env, load_agent_model, save_agent_model  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %( message)s')
logger = logging.getLogger(__name__)

class AlphaZeroAgent(Agent):
    # Add logger as a class attribute
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        action_dim,
        state_dim=2,  # Updated default value
        use_gpu=False,
        num_simulations=800,
        c_puct=1.4,
        load_model=True  # Add this parameter
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_model_flag = load_model  # Store the flag
        self.num_simulations = num_simulations  # Initialize num_simulations
        self.c_puct = c_puct  # Initialize c_puct
        self.current_player = 1  # Initialize current_player
        self.memory = []  # Initialize empty memory list
        self.model = Connect4Net(state_dim, action_dim)
        if not isinstance(self.model, MagicMock):
            self.model = self.model.to(self.device)
        
        # Initialize the logger instance
        self.logger = self.__class__.logger

        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        if load_model:
            load_agent_model(self)  # Use utility function

    def log_gpu_stats(self):
        """Log detailed GPU statistics."""
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            allocated = torch.cuda.memory_allocated()/1e9
            reserved = torch.cuda.memory_reserved()/1e9
            max_memory = torch.cuda.max_memory_allocated()/1e9
            
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
            logger.info(f"GPU Memory - Peak: {max_memory:.2f} GB")
            
            # Reset peak stats
            torch.cuda.reset_peak_memory_stats()

    def preprocess(self, board):
        """
        Preprocesses the board state for the neural network.
        Creates a 2-channel tensor: one for the current player and one for the opponent.

        :param board: Current game board as a NumPy array.
        :return: Preprocessed board as a Torch tensor with shape [2, 6, 7].
        """
        board = board.copy()  # Add this line to prevent modifying the original board
        opponent_player = 2 if self.current_player == 1 else 1
        current_board = (board == self.current_player).astype(float)
        opponent_board = (board == opponent_player).astype(float)
        tensor_board = torch.FloatTensor([current_board, opponent_board]).to(self.device)  # Ensure Tensor type
        return tensor_board

    def load_model(self):
        """Load the agent's model."""
        logger.warning("load_model method is deprecated. Use load_agent_model from utils.py instead.")
        load_agent_model(self)

    def save_model(self):
        """Save the agent's model."""
        save_agent_model(self)  # Removed path parameter

    def select_move(self, game: ConnectFourGame):
        """
        Selects a move using MCTS after ensuring the model is loaded.

        :param game: Instance of ConnectFourGame representing the current game state.
        :return: Selected action (column number).
        """
        if self.load_model_flag and not self.model_loaded:
            self.load_model()

        start_time = time.time()
        # Perform MCTS to get action probabilities
        selected_action, actions, action_probs = self.act(game.board, game, self.num_simulations)
        end_time = time.time()
        logger.info(f"Time taken for select_move: {end_time - start_time:.4f} seconds")

        if selected_action is None:
            logger.warning("No possible actions available.")
            return None

        return selected_action, action_probs  # Return both selected_action and action_probs

    def act(self, state, env, num_simulations=800):
        """
        Selects an action based on the current state using MCTS.

        Parameters:
        - state: Current game state.
        - env: Current game environment.
        - num_simulations: Number of MCTS simulations to perform.

        Returns:
        - selected_action: The chosen action.
        - actions: List of possible actions.
        - action_probs: Probability distribution over actions.
        """
        selected_action, actions, probabilities = self.mcts_simulate(state, env, num_simulations)

        if selected_action is None:
            logger.warning("No valid action selected by MCTS.")
            return None, [], []

        return selected_action, actions, probabilities

    def mcts_simulate(self, state, env, num_simulations=800):
        """
        Simulates MCTS to explore possible actions and their probabilities.

        Parameters:
        - state: Current game state.
        - env: Current game environment.
        - num_simulations: Number of MCTS simulations to perform.

        Returns:
        - selected_action: The chosen action.
        - actions: List of possible actions.
        - action_probs: Probability distribution over actions.
        """
        # Initialize the root node
        root = MCTSNode(parent=None, action=None, env=env.copy())
        root.visits = 1

        for simulation in range(num_simulations):
            node = root
            env_copy = env.copy()
            game_copy = ConnectFourGame()
            game_copy.board = deepcopy(env_copy.board)
            game_copy.current_player = env_copy.current_player

            # === Selection ===
            while not node.is_leaf() and not game_copy.is_terminal():
                node = node.best_child(c_puct=self.c_puct)
                if node is None:
                    break
                game_copy.make_move(node.action)
                # Switch player
                game_copy.current_player = 2 if game_copy.current_player == 1 else 1

            if node is None:
                continue

            # === Evaluation ===
            if game_copy.is_terminal():
                # Terminal node reached; get reward and backpropagate
                reward = game_copy.get_reward()
                node.backpropagate(reward)
                continue

            # === Expansion ===
            legal_moves = game_copy.get_valid_locations()
            if not legal_moves:
                # No moves possible
                node.backpropagate(0)
                continue

            state_tensor = self.preprocess(game_copy.board).unsqueeze(0)
            # Ensure that game_copy.board remains a NumPy array after preprocessing
            with torch.no_grad():
                log_policy, value = self.model(state_tensor)
                policy = torch.exp(log_policy).cpu().numpy().flatten()

            # Mask illegal moves
            filtered_probs = np.zeros(self.action_dim)
            filtered_probs[legal_moves] = policy[legal_moves]
            if filtered_probs.sum() > 0:
                filtered_probs /= filtered_probs.sum()
            else:
                # Assign uniform probabilities if all probabilities are zero
                filtered_probs[legal_moves] = 1.0 / len(legal_moves)

            node.expand(filtered_probs, legal_moves)

            # === Simulation ===
            node_value = value.item()
            # Backpropagate the value from the neural network
            node.backpropagate(node_value)

        # After simulations, select the action with the highest visit count
        action_visits = [(child.action, child.visits) for child in root.children.values()]
        if not action_visits:
            logger.error("No actions available after MCTS simulations.")
            return None, [], []

        selected_action = max(action_visits, key=lambda x: x[1])[0]

        # Compute the action probabilities
        total_visits = sum(child.visits for child in root.children.values())
        action_probs = np.zeros(self.action_dim)
        for child in root.children.values():
            action_probs[child.action] = child.visits / total_visits

        # Convert action_probs to a FloatTensor before returning
        action_probs = torch.FloatTensor(action_probs).to(self.device)

        return selected_action, list(root.children.keys()), action_probs

    def self_play(self):
        """
        Conducts a self-play game and stores the game data.
        """
        game_data = []
        state = self.env.reset()
        done = False
        player = 1  # Player 1 starts

        while not done:
            self.current_player = player
            selected_action, action_probs = self.select_move(self.env)
            if selected_action is None:
                logger.error("Agent failed to select a valid action during self-play.")
                break

            # action_probs is already a FloatTensor from select_move

            game_data.append((state.copy(), action_probs, player))
            state, reward, done, _ = self.env.step(selected_action)
            player = -player  # Switch players

        winner = self.env.get_winner()
        logger.info(f"Game ended. Winner: {winner}")

        # Assign value targets based on the winner
        for state_data, mcts_prob, player_in_game in game_data:
            if winner == 0:
                value = 0  # Draw
            else:
                value = 1 if winner == player_in_game else -1
            self.memory.append((state_data, mcts_prob, value))

    # Removed train_step since training is handled by LightningModule

    def perform_training(self):
        """Perform training steps for the agent."""
        from .train.train_alpha_zero import train_alphazero  # Moved import here to prevent circular dependency
        train_alphazero(time_limit=3600, num_self_play_games=1000, use_gpu=self.device.type == 'cuda', load_model=self.model_loaded)

__all__ = ['AlphaZeroAgent']