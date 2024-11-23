# /agent_code.py

import logging
from collections import deque
import os
import time
import numpy as np
import torch
import copy

from nnbattle.game.connect_four_game import ConnectFourGame
from .network import Connect4Net
from .utils.model_utils import load_agent_model, save_agent_model, MODEL_PATH
from .mcts import MCTSNode
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
        load_model=True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_model_flag = load_model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.current_player = 1
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
        opponent_player = 2 if self.current_player == 1 else 1
        current_board = (board == self.current_player).astype(float)
        opponent_board = (board == opponent_player).astype(float)
        tensor_board = torch.FloatTensor([current_board, opponent_board]).to(self.device)
        return tensor_board

    def load_model_method(self):
        logger.warning("load_model method is deprecated. Use load_agent_model from utils.py instead.")
        load_agent_model(self)

    def save_model_method(self):
        save_agent_model(self, self.model_path)  # Pass the model_path explicitly

    def select_move(self, game: ConnectFourGame):
        """Select a move and return action probabilities."""
        if self.load_model_flag and not self.model_loaded:
            try:
                load_agent_model(self)
                self.model_loaded = True
            except FileNotFoundError:
                logger.warning("No model file found, using untrained model.")
                self.model_loaded = False

        start_time = time.time()
        selected_action, action_probs = self.act(game)
        end_time = time.time()
        logger.info(f"Time taken for select_move: {end_time - start_time:.4f} seconds")

        if selected_action is None:
            logger.warning("No possible actions available.")
            # Always return a tensor for action_probs
            return None, torch.zeros(self.action_dim, device=self.device)

        return selected_action, action_probs

    def act(self, game: ConnectFourGame):
        selected_action, action_probs = self.mcts_simulate(game)
        if selected_action is None:
            logger.warning("No valid action selected by MCTS.")
            return None, []

        return selected_action, action_probs

    def mcts_simulate(self, game: ConnectFourGame):
        root = MCTSNode(parent=None, action=None, env=deepcopy_env(game))
        root.visits = 1

        for _ in range(self.num_simulations):
            node = root
            game_copy = deepcopy_env(game)

            while not node.is_leaf() and not game_copy.is_terminal():
                node = node.best_child(c_puct=self.c_puct)
                if node is None:
                    break
                game_copy.make_move(node.action)
                game_copy.current_player = 2 if game_copy.current_player == 1 else 1

            if node is None:
                continue

            if game_copy.is_terminal():
                reward = game_copy.get_reward()
                node.backpropagate(reward)
                continue

            legal_moves = game_copy.get_valid_locations()
            if not legal_moves:
                node.backpropagate(0)
                continue

            state_tensor = self.preprocess(game_copy.board).unsqueeze(0)
            with torch.no_grad():
                log_policy, value = self.model(state_tensor)
                policy = torch.exp(log_policy).cpu().numpy().flatten()

            filtered_probs = np.zeros(self.action_dim)
            filtered_probs[legal_moves] = policy[legal_moves]
            if (filtered_probs.sum() > 0):
                filtered_probs /= filtered_probs.sum()
            else:
                filtered_probs[legal_moves] = 1.0 / len(legal_moves)

            node.expand(filtered_probs, legal_moves)
            node.backpropagate(value.item())

        action_visits = [(child.action, child.visits) for child in root.children.values()]
        if not action_visits:
            logger.error("No actions available after MCTS simulations.")
            return None, torch.zeros(self.action_dim, device=self.device)

        selected_action = max(action_visits, key=lambda x: x[1])[0]
        total_visits = sum(child.visits for child in root.children.values())
        action_probs = torch.zeros(self.action_dim, device=self.device)
        for child in root.children.values():
            action_probs[child.action] = child.visits / total_visits

        return selected_action, action_probs

    def self_play(self, max_moves=100):
        game_data = []
        game = ConnectFourGame()
        player = 1
        move_count = 0

        while not game.is_terminal() and move_count < max_moves:
            self.current_player = player
            selected_action, action_probs = self.select_move(game)
            if selected_action is None:
                logger.error("Agent failed to select a valid action during self-play.")
                break

            game_data.append((game.get_state().copy(), action_probs.numpy(), player))
            game.make_move(selected_action)
            player = -player  # Switch player
            move_count += 1

            # Add logging to track the game state
            logger.info(f"Move {move_count}: {selected_action}, Current player: {player}")
            logger.info(f"Board state:\n{game.board_to_string()}")

        if move_count >= max_moves:
            logger.warning("Reached maximum number of moves without a terminal state.")

        winner = game.get_winner()
        logger.info(f"Game ended. Winner: {winner}")

        for state_data, mcts_prob, player_in_game in game_data:
            if winner == 0:
                value = 0
            else:
                value = 1 if winner == player_in_game else -1
            self.memory.append((state_data, mcts_prob, value))

    def perform_training(self):
        """Perform training using train_alphazero."""
        # Explicitly import from trainer module
        from nnbattle.agents.alphazero.train.trainer import train_alphazero
        
        train_alphazero(
            time_limit=3600,
            num_self_play_games=1000,
            use_gpu=self.device.type == 'cuda',
            load_model=self.model_loaded
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