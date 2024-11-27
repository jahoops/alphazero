import numpy as np
import copy
from .mcts import mcts_simulate
from typing import List, Tuple
from ...utils.logger_config import logger
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class SelfPlay:
    def __init__(self, game: ConnectFourGame, model, num_simulations=100, agent=None):
        """Initialize with game instance and agent."""
        if not isinstance(game, ConnectFourGame):
            raise TypeError("game must be an instance of ConnectFourGame")
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.agent = agent  # Store the agent reference

    def execute_episode(self, game_num) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Execute one episode of self-play using MCTS."""
        # Create fresh game instance for this episode
        game = ConnectFourGame()  # Each episode gets its own game instance
        states, policies, values = [], [], []
        game_history = []
        current_team = RED_TEAM
        
        while game.get_game_state() == "ONGOING":
            try:
                # Get move from MCTS (which will create its own game instances)
                action, policy = mcts_simulate(self.agent, game, current_team)
                
                # Store current state and policy
                game_history.append((game.get_board().copy(), policy, current_team))
                
                # Apply move to main game instance
                if not game.make_move(action, current_team):
                    break
                    
                # Switch teams only after successful move
                current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM
                
            except Exception as e:
                logger.error(f"Error in self-play game {game_num}: {e}")
                break

        # Process game history with proper rewards
        result = game.get_game_state()
        if result == "ONGOING":
            return [], [], []
            
        for state, policy, team in reversed(game_history):
            reward = np.float32(0.0 if result == "Draw" else (1.0 if result == team else -1.0))  # Ensure float32
            states.append(state)
            policies.append(policy)
            values.append(reward)
            
        return states, policies, values

    def get_mcts_policy(self, state) -> np.ndarray:
        """Get policy from MCTS search using valid moves from the game."""
        valid_moves = self.game.get_valid_moves()  # Get valid moves from the game instance
        # Create a policy array of zeros
        policy = np.zeros(7, dtype=np.float32)  # Ensure float32
        # Set equal probability for valid moves
        if len(valid_moves) > 0:
            policy[valid_moves] = 1.0 / len(valid_moves)
        return policy

    def generate_training_data(self, num_episodes: int) -> List[Tuple]:
        """Generate training data through self-play."""
        training_data = []
        total_moves = 0
        completed_games = 0
        
        for game_num in range(num_episodes):
            states, policies, values = self.execute_episode(game_num)
            if states:  # Only count completed games
                completed_games += 1
                total_moves += len(states)
                training_data.extend(zip(states, policies, values))
        
        avg_moves = total_moves / completed_games if completed_games > 0 else 0
        logger.info(f"Total training examples generated: {len(training_data)}")
        logger.info(f"Average moves per game: {avg_moves:.2f} over {completed_games} completed games")
        return training_data