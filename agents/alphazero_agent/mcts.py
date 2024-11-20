# alphazero_agent/mcts.py

import math
import logging
from copy import deepcopy

from .utils import deepcopy_env  # Assuming you have a deepcopy utility
from game.connect_four_game import ConnectFourGame  # Update the import path as needed

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, parent=None, action=None, env=None):
        self.parent = parent          # Parent node
        self.action = action          # Action taken to reach this node
        self.children = {}            # Dictionary to store child nodes {action: MCTSNode}
        self.visits = 0               # Number of times the node was visited
        self.value_sum = 0.0          # Total value of the node
        self.prior = 0.0              # Prior probability of selecting this action
        self.env = deepcopy(env) if env else None  # Game environment/state

    def is_leaf(self):
        """
        Determines if the node is a leaf node (no children).
        """
        return len(self.children) == 0

    def best_child(self, c_puct=1.0):
        """
        Selects the best child node based on the PUCT formula.

        :param c_puct: Exploration parameter.
        :return: Best child node.
        """
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
        """
        Expands the node by creating child nodes for each legal action.

        :param action_probs: Array of action probabilities.
        :param legal_moves: List of legal actions.
        """
        for action in legal_moves:
            if action not in self.children:
                new_env = deepcopy(self.env)
                new_env.make_move(action)
                child_node = MCTSNode(parent=self, action=action, env=new_env)
                child_node.prior = action_probs[action]
                self.children[action] = child_node
                logger.debug(f"Expanded node with action {action} and prior {child_node.prior:.4f}.")

    def backpropagate(self, value):
        """
        Backpropagates the value up the tree.

        :param value: The value to backpropagate.
        """
        self.visits += 1
        self.value_sum += value
        if self.parent:
            # Negate the value to switch perspectives between players
            self.parent.backpropagate(-value)