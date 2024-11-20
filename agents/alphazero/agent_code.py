# /agent_code.py

import logging
from collections import deque

import numpy as np
import torch

from game.connect_four_game import \
    ConnectFourGame  # Update the import path as needed

from .mcts import MCTSNode
from .network import Connect4Net

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AlphaZeroAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        use_gpu=False,
        model_path="alphazero/model/alphazero_model_final.pth",
        num_simulations=800,
        c_puct=1.4
    ):
        """
        Initializes the AlphaZeroAgent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model = Connect4Net(state_dim, action_dim).to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        self.env = ConnectFourGame()  # Initialize the game environment/state
        self.model_path = model_path
        self.model_loaded = False  # Flag to check if model is loaded
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Memory for storing training samples
        self.memory = deque(maxlen=10000)  # Use deque for efficient FIFO operations

        # Initialize current_player
        self.current_player = 1  # Player 1 starts by default

    def preprocess(self, board):
        """
        Preprocesses the board state for the neural network.
        Multiplies the board by the current player to get the perspective.

        :param board: Current game board as a NumPy array.
        :return: Preprocessed board as a Torch tensor.
        """
        tensor_board = torch.FloatTensor(board * self.current_player).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor_board

    def load_model(self):
        """
        Loads model weights from the specified path.
        """
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise e

    def save_model(self, path):
        """
        Saves model weights to the specified path.

        :param path: Destination path for the model weights.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}.")

    def select_move(self, game: ConnectFourGame):
        """
        Selects a move using MCTS after ensuring the model is loaded.

        :param game: Instance of ConnectFourGame representing the current game state.
        :return: Selected action (column number).
        """
        if not self.model_loaded:
            self.load_model()

        # Perform MCTS to get action probabilities
        selected_action, actions, probs = self.act(game.board, game, self.num_simulations)

        if selected_action is None:
            logger.warning("No possible actions available.")
            return None

        logger.info(f"Selected action: {selected_action}")
        return selected_action

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
        # Perform MCTS simulations
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

            # === Selection ===
            while not node.is_leaf() and not env_copy.is_terminal():
                node = node.best_child(c_puct=self.c_puct)
                if node is None:
                    break
                env_copy.make_move(node.action)
                self.current_player = -self.current_player  # Switch player

            if node is None:
                continue

            # === Evaluation ===
            if env_copy.is_terminal():
                # Terminal node reached; get reward and backpropagate
                reward = env_copy.get_reward()
                node.backpropagate(reward)
                continue

            # === Expansion ===
            legal_moves = env_copy.get_valid_locations()
            if not legal_moves:
                # No moves possible
                node.backpropagate(0)
                continue

            state_tensor = self.preprocess(env_copy.board)
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

        return selected_action, list(root.children.keys()), action_probs.tolist()

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
            selected_action = self.select_move(self.env)
            if selected_action is None:
                logger.error("Agent failed to select a valid action during self-play.")
                break

            # Placeholder for action probabilities; in practice, use MCTS probabilities
            action_probs = [0.0] * self.action_dim
            action_probs[selected_action] = 1.0  # Assuming deterministic selection for self-play

            game_data.append((state.copy(), action_probs.copy(), player))
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