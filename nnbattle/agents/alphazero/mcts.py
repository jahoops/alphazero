# /mcts.py

import logging
import math
import copy
from typing import Optional, List

import torch
import numpy as np  # Add numpy import if not already present
from nnbattle.game import ConnectFourGame, InvalidMoveError
from nnbattle.constants import RED_TEAM, YEL_TEAM
from nnbattle.agents.alphazero.utils.model_utils import preprocess_board

logger = logging.getLogger(__name__)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], env: ConnectFourGame):
        self.parent = parent
        self.action = action
        # Determine the team that will make the next move
        if env.last_team is None:
            self.team = RED_TEAM  # Starting team
        else:
            self.team = 3 - env.last_team  # Switch team
        self.env = env
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
        for action in legal_actions:
            if action not in self.children:
                new_env = deepcopy_env(self.env)
                new_env.make_move(action, self.team)  # self.team is now valid
                child_node = MCTSNode(parent=self, action=action, env=new_env)
                child_node.prior = action_probs[action].item()
                self.children[action] = child_node

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

def mcts_simulate(agent, game: ConnectFourGame, valid_moves, temperature=1.0):
    logger.info("Starting MCTS simulation...")
    root = MCTSNode(parent=None, action=None, env=deepcopy_env(game))
    root.visits = 1

    # Add Dirichlet noise to the root node's priors
    action_probs = np.ones(agent.action_dim) / agent.action_dim  # Uniform prior
    dirichlet_alpha = 0.3  # Adjust alpha as needed
    dirichlet_noise = np.random.dirichlet([dirichlet_alpha] * agent.action_dim)
    action_probs = 0.75 * action_probs + 0.25 * dirichlet_noise
    valid_actions = game.get_valid_moves()
    action_probs = torch.tensor(action_probs, dtype=torch.float32, device=agent.device)  # Ensure on correct device
    # Mask invalid moves
    action_mask = torch.zeros(agent.action_dim, dtype=torch.float32, device=agent.device)
    action_mask[valid_actions] = 1
    action_probs *= action_mask
    if action_probs.sum() > 0:
        action_probs /= action_probs.sum()
    else:
        # If all probabilities are zero, assign equal probability to valid actions
        action_probs[valid_actions] = 1.0 / len(valid_actions)
    root.expand(action_probs, valid_actions)

    for simulation in range(agent.num_simulations):
        node = root
        env = deepcopy_env(game)
        logger.debug(f"Simulation {simulation + 1}/{agent.num_simulations} started.")

        # **Selection**
        while not node.is_leaf():
            node = node.best_child(agent.c_puct)
            if node is None:
                logger.debug("No child nodes available during selection.")
                break
            if env.last_team is None:
                env.make_move(node.action, RED_TEAM)
            else:
                env.make_move(node.action, 3 - env.last_team)
            logger.debug(f"Moved to child node: Team {node.team}, Action {node.action}")

        # **Expansion**
        if env.get_game_state() == "ONGOING":
            with torch.cuda.device(agent.device):
                # Ensure state tensor and model are on same device
                model_device = next(agent.model.parameters()).device
                state = agent.preprocess(env.get_board()).to(model_device)
                with torch.no_grad():
                    agent.model.eval()  # Set model to eval mode
                    action_probs, value = agent.model(state.unsqueeze(0))
                    # Extract scalar value from tensor
                    value = value.squeeze().item()  # Fix: properly extract scalar value
                    action_probs = action_probs.squeeze().detach()

            # Mask invalid moves
            valid_actions = env.get_valid_moves()
            action_mask = torch.zeros(agent.action_dim, device=agent.device)
            action_mask[valid_actions] = 1
            action_probs *= action_mask
            if action_probs.sum() > 0:
                action_probs /= action_probs.sum()
            else:
                # If all probabilities are zero, assign equal probability to valid actions
                action_probs[valid_actions] = 1.0 / len(valid_actions)
            node.expand(action_probs, valid_actions)
            reward = value  # **Use the network's value prediction as reward**
            logger.debug(f"Assigned reward {reward} to the node based on network prediction.")

        else:
            # **Terminal State**
            final_state = env.get_game_state()
            if final_state == RED_TEAM or final_state == YEL_TEAM:
                reward = 1.0 if final_state == agent.team else -1.0
                logger.debug(f"Terminal state: Team {final_state} wins with reward {reward}")
            else:
                reward = 0.0  # Draw
                logger.debug(f"Terminal state: Draw with reward {reward}")

        # **Simulation and Backpropagation**
        node.backpropagate(reward)
        logger.debug(f"Backpropagated reward {reward} to node.")

    # **Selection of the Best Action with Temperature**
    action_visits = [(child.action, child.visits) for child in root.children.values()]
    if not action_visits:
        logger.error("No valid moves found during MCTS.")
        raise InvalidMoveError("No valid moves found during MCTS.")

    actions, visits = zip(*action_visits)
    visits = np.array(visits, dtype=np.float32)

    if temperature == 0:
        selected_action = actions[np.argmax(visits)]
        action_probs = np.zeros(agent.action_dim, dtype=np.float32)
        action_probs[selected_action] = 1.0
    else:
        probs = visits ** (1 / temperature)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones_like(visits) / len(visits)  # Fix: use visits length for proper shape
        selected_action = np.random.choice(actions, p=probs)
        action_probs = np.zeros(agent.action_dim, dtype=np.float32)  # Fix: initialize full array
        action_probs[list(actions)] = probs  # Fix: assign probabilities to correct indices

    logger.info(f"MCTS selected action {selected_action} with action_probs {action_probs}")
    return selected_action, torch.tensor(action_probs, dtype=torch.float32, device=agent.device)

__all__ = ['MCTSNode', 'mcts_simulate']

# No changes needed if not mocking base classes
# Ensure that in tests, only methods are mocked, not entire classes that use super()