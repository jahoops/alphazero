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
        self.team = 1
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
            return None, torch.zeros(self.action_dim, device=self.device)

        return selected_action, action_probs

    def mcts_simulate(self, game: ConnectFourGame):
        root = MCTSNode(parent=None, action=None, env=deepcopy_env(game))
        root.visits = 1

        for _ in range(self.num_simulations):
            node = root
            game_copy = deepcopy_env(game)

            while not node.is_leaf() and game_copy.get_game_state() == "ONGOING":
                node = node.best_child(c_puct=self.c_puct)
                if node is None:
                    break
                game_copy.make_move(node.action, self.team)
                self.team = 2 if self.team == 1 else 1

            if node is None:
                continue

            state = game_copy.get_game_state()
            if state in ["RED_WINS", "YEL_WINS", "DRAW"]:
                if state == "RED_WINS":
                    reward = 1 if game_copy.last_piece == RED_PIECE else -1
                elif state == "YEL_WINS":
                    reward = 1 if game_copy.last_piece == YEL_PIECE else -1
                else:
                    reward = 0
                node.backpropagate(reward)
                continue

            legal_moves = game_copy.get_valid_moves()
            if not legal_moves:
                node.backpropagate(0)
                continue

            state_tensor = self.preprocess(game_copy.get_board()).unsqueeze(0)
            with torch.no_grad():
                log_policy, value = self.model(state_tensor)
                policy = torch.exp(log_policy).cpu().numpy().flatten()

            filtered_probs = np.zeros(self.action_dim)
            filtered_probs[legal_moves] = policy[legal_moves]
            if filtered_probs.sum() > 0:
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

        # Ensure all code paths return two values
        return selected_action, action_probs

    def self_play(self, max_moves=100):
        game_data = []
        game = ConnectFourGame()
        player = 1
        move_count = 0

        while game.get_game_state() == "ONGOING" and move_count < max_moves:
            self.team = player
            selected_action, action_probs = self.select_move(game)
            if selected_action is None:
                logger.error("Agent failed to select a valid action during self-play.")
                break

            preprocessed_state = self.preprocess(game.get_board())

            game_data.append((
                preprocessed_state.cpu().numpy(),
                action_probs.cpu().numpy(),
                player
            ))
            game.make_move(selected_action, player)
            player = 2 if player == 1 else 1
            move_count += 1

            logger.info(f"Move {move_count}: {selected_action}, Current player: {player}")
            logger.info(f"Board state:\n{game.board_to_string()}")

        if move_count >= max_moves:
            logger.warning("Reached maximum number of moves without a terminal state.")

        state, _, player_in_game = game_data[-1]
        state_last = game.get_game_state()
        if state_last == "RED_WINS":
            value = 1 if player_in_game == RED_PIECE else -1
        elif state_last == "YEL_WINS":
            value = 1 if player_in_game == YEL_PIECE else -1
        else:
            value = 0

        for state_data, mcts_prob, player_in_game in game_data:
            self.memory.append((state_data, mcts_prob, value))

    def perform_training(self):
        """Perform training using train_alphazero.

        :param max_iterations: Maximum number of training iterations.
        """
        from nnbattle.agents.alphazero.train.trainer import train_alphazero

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