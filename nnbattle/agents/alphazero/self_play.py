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

    def execute_episode(self, episode_num: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Execute one episode of self-play."""
        logger.info(f"Starting episode {episode_num} execution")
        game = ConnectFourGame()
        states, policies, values = [], [], []
        game_history = []
        current_team = RED_TEAM
        move_count = 0
        
        try:
            while game.get_game_state() == "ONGOING":
                logger.info(f"Episode {episode_num} - Move {move_count + 1}, Team {current_team}")
                
                # Log game state for debugging
                logger.debug(f"Current board state:\n{game.board_to_string()}")
                
                # Get valid moves
                valid_moves = game.get_valid_moves()
                logger.debug(f"Valid moves: {valid_moves}")
                
                if not valid_moves:
                    logger.error("No valid moves available")
                    break
                
                try:
                    # Perform MCTS simulation with timeout protection
                    action, policy = mcts_simulate(self.agent, game, current_team)
                    logger.info(f"Selected action: {action}")
                    
                    # Make move
                    if not game.make_move(action, current_team):
                        logger.error(f"Invalid move {action} for team {current_team}")
                        break
                    
                    # Store state and policy
                    game_history.append((
                        game.get_board().copy(),
                        policy,
                        current_team
                    ))
                    
                    # Switch teams
                    current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM
                    move_count += 1
                    
                except Exception as e:
                    logger.error(f"Error during move {move_count}: {str(e)}")
                    raise
                
            # Process game result
            result = game.get_game_state()
            logger.info(f"Episode {episode_num} finished with result: {result} after {move_count} moves")
            
            # Process game history
            for state, policy, team in game_history:
                reward = 0.0 if result == "Draw" else (1.0 if result == team else -1.0)
                states.append(state)
                policies.append(policy)
                values.append(reward)
            
            return states, policies, values
            
        except Exception as e:
            logger.error(f"Episode {episode_num} failed: {str(e)}")
            return [], [], []

    def _process_game_history(self, game_history: List[Tuple], result: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Process game history into training data."""
        if not game_history:
            return [], [], []
            
        states, policies, values = [], [], []
        for state, policy, team in reversed(game_history):
            reward = 0.0 if result == "Draw" else (1.0 if result == team else -1.0)
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
        logger.info(f"Starting self-play data generation for {num_episodes} episodes")
        training_data = []
        completed_games = 0
        
        try:
            for episode in range(num_episodes):
                if hasattr(self, '_interrupt_requested'):
                    logger.info("Interrupt requested, saving current progress...")
                    break
                    
                logger.info(f"Starting episode {episode + 1}/{num_episodes}")
                try:
                    states, policies, values = self.execute_episode(episode)
                    if states:
                        training_data.extend(zip(states, policies, values))
                        completed_games += 1
                        
                    if episode % 5 == 0:
                        logger.info(f"Progress: {episode + 1}/{num_episodes} episodes, "
                                  f"Examples: {len(training_data)}")
                        
                except KeyboardInterrupt:
                    logger.info("Interrupt received during episode, saving progress...")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupt received, saving current progress...")
        finally:
            logger.info(f"Self-play ended with {len(training_data)} examples "
                       f"from {completed_games} games")
            return training_data