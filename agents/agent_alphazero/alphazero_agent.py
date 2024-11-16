# alphazero_agent.py

from connect4 import Connect4  # Import only Connect4
from torch.utils.data import DataLoader  # Add DataLoader import
from collections import deque  # Ensure deque is imported
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import numpy as np
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Ensure no in-place operations
        out = self.relu(out)
        return out

class Connect4Net(nn.Module):
    def __init__(self, state_dim, action_dim, num_res_blocks=5, num_filters=128):
        super(Connect4Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=4, padding=2, stride=2),  # Output: (num_filters, 4, 4)
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, padding=2, stride=2),  # (num_filters*2, 2, 2)
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=4, padding=2, stride=2),  # (num_filters*2, 1,1)
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters * 2) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters * 2, 2, kernel_size=1),  # Output: (2, 1, 1)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (batch_size, 2)
            nn.Linear(8, action_dim),  # Updated in_features from 2 to 8
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters * 2, 1, kernel_size=1),  # Output: (1, 1, 1)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (batch_size, 4)
            nn.Linear(4, 256),  # Updated in_features from 1 to 4
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.residual_blocks(x)
        #print(f"After Conv and Residuals: {x.shape}")  # Debug statement
        policy = self.policy_head(x)
        value = self.value_head(x)
        #print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")  # Debug statement
        return F.log_softmax(policy, dim=1), value

class MCTSNode:
    def __init__(self, parent=None, action=None, env=None):
        self.parent = parent          # Parent node
        self.action = action          # Action taken to reach this node
        self.children = {}            # Dictionary to store child nodes {action: MCTSNode}
        self.visits = 0               # Number of times the node was visited
        self.value_sum = 0.0          # Total value of the node
        self.prior = 0.0              # Prior probability of selecting this action
        self.env = env                # Game environment/state

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_child = None
        sqrt_total_visits = math.sqrt(self.visits)

        for child in self.children.values():
            if child.visits == 0:
                u = c_puct * child.prior * sqrt_total_visits
            else:
                u = c_puct * child.prior * sqrt_total_visits / (1 + child.visits)
            q = child.value_sum / child.visits if child.visits > 0 else 0
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, action_probs, legal_moves):
        for action in legal_moves:
            if action not in self.children:
                new_env = deepcopy(self.env)
                new_env.make_move(action)
                child_node = MCTSNode(parent=self, action=action, env=new_env)
                child_node.prior = action_probs[action]
                self.children[action] = child_node
                logger.debug(f"Expanded node with action {action} and prior {child_node.prior:.4f}.")

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            # Negate the value to switch perspectives between players
            self.parent.backpropagate(-value)

class AlphaZeroAgent:
    def __init__(self, state_dim, action_dim, use_gpu=False, model_path="alphazero_agent/model/alphazero_model_final.pth"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model = Connect4Net(state_dim, action_dim).to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        self.env = Connect4()  # Initialize the game environment/state
        self.model_path = model_path
        self.model_loaded = False  # Flag to check if model is loaded

        # Initialize optimizer and loss functions for training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()

        # Memory for storing training samples
        self.memory = deque(maxlen=10000)  # Use deque for efficient FIFO operations

        # Initialize current_player
        self.current_player = 1  # Player 1 starts by default

    def preprocess(self, board):
        """
        Preprocesses the board state for the neural network.
        Multiplies the board by the current player to get the perspective.
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
        selected_action, actions, probs = self.act(game.board, game)

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
        # Initialize the root node
        root = MCTSNode(parent=None, action=None, env=env.copy())
        root.visits = 1

        for simulation in range(num_simulations):
            node = root
            env_copy = env.copy()

            # === Selection ===
            while not node.is_leaf() and not env_copy.is_terminal():
                node = node.best_child(c_puct=1.0)
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

    def train_step(self, batch_size=32):
        """
        Perform a single training step using data from self-play games.
        """
        if len(self.memory) < batch_size:
            logger.debug("Not enough samples in memory to perform a training step.")
            return

        # Sample a batch from memory without replacement
        batch = random.sample(self.memory, batch_size)
        states, mcts_probs, values = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        log_policy, predicted_value = self.model(states)

        # Compute losses
        value_loss = self.criterion_value(predicted_value.squeeze(), values)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_policy, dim=1))
        loss = value_loss + policy_loss

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        logger.debug(f"Training step completed with loss {loss.item():.4f}.")

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