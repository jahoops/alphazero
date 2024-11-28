# /mcts.py

import logging
import math
import copy
from typing import Optional, List
import signal
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
import numpy as np
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM
from .utils.model_utils import preprocess_board

logger = logging.getLogger(__name__)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], board: np.ndarray, team: int):
        """Initialize node with board state instead of game instance."""
        self.parent = parent
        self.action = action
        self.board = board.copy()  # Store board state instead of game instance
        self.team = team
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c_puct: float) -> Optional['MCTSNode']:
        best_score = -float('inf')
        best_node = None
        for child in self.children.values():
            u = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            q = child.value / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self, action_probs: torch.Tensor, legal_actions: List[int]):
        """
        Expand node with valid moves from the game state.
        """
        # Get valid moves directly from the environment
        valid_moves = self.env.get_valid_moves()
        for action in valid_moves:  # Only iterate over valid moves
            if action not in self.children:
                new_env = deepcopy_env(self.env)
                try:
                    # Make move with current team
                    new_env.make_move(action, self.team)
                    # Next team is opposite of last_team
                    next_team = YEL_TEAM if new_env.last_team == RED_TEAM else RED_TEAM
                    child_node = MCTSNode(parent=self, action=action, env=new_env, team=next_team)
                    child_node.prior = action_probs[action].item()
                    self.children[action] = child_node
                except (InvalidMoveError, InvalidTurnError) as e:
                    logger.error(f"Invalid move during expansion: {e}")
                    continue

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

def mcts_simulate(agent, game: ConnectFourGame, team: int, temperature=1.0):
    """Monte Carlo Tree Search simulation."""
    logger.debug(f"Starting MCTS simulation for team {team}")
    
    try:
        if game.last_team == team:
            raise InvalidTurnError(f"Invalid turn: team {team} cannot move after itself")

        # Create a fresh game instance for MCTS
        root_game = ConnectFourGame()
        root_game.board = game.get_board().copy()
        root_game.last_team = game.last_team  # Copy the last_team state

        # Initialize root with root_game
        root = MCTSNode(None, None, root_game.get_board(), team)
        root.visits = 1

        # Run simulations
        for sim in range(agent.num_simulations):
            if sim % 10 == 0:  # Log progress every 10 simulations
                logger.debug(f"Simulation {sim + 1}/{agent.num_simulations}")
            
            # Create new game instance for this simulation path
            sim_game = ConnectFourGame()
            sim_game.board = root_game.get_board().copy()
            sim_game.last_team = root_game.last_team
            
            node = root
            current_team = team

            # Selection
            while not node.is_leaf():
                node = node.best_child(agent.c_puct)
                if node is None:
                    break
                # Apply action to simulation game
                sim_game.make_move(node.action, current_team)
                current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM

            # Expansion and evaluation
            if sim_game.get_game_state() == "ONGOING" and node.is_leaf():
                sim_valid_moves = sim_game.get_valid_moves()
                if sim_valid_moves:
                    # Create new game instances for each child
                    for move in sim_valid_moves:
                        child_game = ConnectFourGame()
                        child_game.board = sim_game.get_board().copy()
                        child_game.last_team = sim_game.last_team
                        if child_game.make_move(move, current_team):
                            child = MCTSNode(
                                parent=node,
                                action=move,
                                board=child_game.get_board(),
                                team=(YEL_TEAM if current_team == RED_TEAM else RED_TEAM)
                            )
                            # Get network predictions
                            with torch.no_grad():
                                state_tensor = agent.preprocess(child_game.get_board(), current_team, to_device=agent.device)
                                action_logits, value = agent.model(state_tensor.unsqueeze(0))
                                action_probs = F.softmax(action_logits.squeeze(), dim=0)
                                child.prior = action_probs[move].item()
                            node.children[move] = child

            # Backpropagation
            game_result = sim_game.get_game_state()
            if game_result == "ONGOING":
                with torch.no_grad():
                    state_tensor = agent.preprocess(sim_game.get_board(), current_team, to_device=agent.device)
                    _, value = agent.model(state_tensor.unsqueeze(0))
                    value = value.item()
            else:
                value = 1.0 if game_result == team else (-1.0 if game_result in (RED_TEAM, YEL_TEAM) else 0.0)
            node.backpropagate(-value)  # Negative because we're alternating perspectives

        # Select move based on visit counts and temperature
        valid_children = [(child.action, child.visits) for child in root.children.values()]
        if not valid_children:
            valid_moves = game.get_valid_moves()
            return np.random.choice(valid_moves), torch.zeros(agent.action_dim)

        actions, visits = zip(*valid_children)
        visits = np.array(visits, dtype=np.float32)

        # Select action based on temperature
        if temperature < 0.01:
            action = actions[np.argmax(visits)]
        else:
            probs = visits ** (1.0 / temperature)
            probs = probs / probs.sum()
            action = np.random.choice(actions, p=probs)

        # Create policy
        policy = torch.zeros(agent.action_dim, dtype=torch.float32)
        policy[list(actions)] = torch.tensor(visits / visits.sum(), dtype=torch.float32)
        
        return action, policy

    except Exception as e:
        logger.error(f"MCTS simulation failed: {str(e)}")
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        return np.random.choice(valid_moves), torch.zeros(agent.action_dim)
