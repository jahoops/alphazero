import numpy as np
from typing import List, Tuple
from ...utils.logger_config import logger
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM

class SelfPlay:
    def __init__(self, game: ConnectFourGame, model, num_simulations=100):
        """Initialize with a ConnectFourGame instance."""
        if not isinstance(game, ConnectFourGame):
            raise TypeError("game must be an instance of ConnectFourGame")
        self.game = game
        self.model = model
        self.num_simulations = num_simulations

    def execute_episode(self, game_num) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Execute one episode of self-play until game completion."""
        states, policies, values = [], [], []
        # Create a new game instance for this episode
        game = ConnectFourGame()  
        game_history = []
        current_team = RED_TEAM
        
        max_moves = 42  # Maximum possible moves in Connect Four (6x7 board)
        move_count = 0
        
        while True:
            # Check if game is actually complete
            game_state = game.get_game_state()
            if game_state != "ONGOING" or move_count >= max_moves:
                logger.info(f"Game {game_num} complete. Result: {game_state}")
                break
                
            # Get valid moves before selecting action
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                logger.warning("No valid moves available but game is ongoing!")
                break
                
            try:
                # Store current state and policy before making move
                current_state = game.get_board().copy()
                policy = np.zeros(7)  # 7 columns in Connect Four
                policy[valid_moves] = 1.0 / len(valid_moves)
                
                # Choose action from valid moves only
                action = np.random.choice(valid_moves)
                game_history.append((current_state, policy, current_team))
                
                # Make the move
                game.make_move(action, current_team)
                move_count += 1
                
                # Switch teams
                current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM
                
            except (InvalidMoveError, InvalidTurnError) as e:
                logger.error(f"Invalid move during self-play: {e}")
                break
        
        # Get final game result
        result = game.get_game_state()
        if result == "ONGOING":
            logger.error("Game ended while still ongoing!")
            return [], [], []  # Return empty lists for invalid game
            
        # Process game history with proper rewards
        for state, policy, team in reversed(game_history):
            if result == "Draw":
                reward = 0.0
            else:
                reward = 1.0 if result == team else -1.0
            states.append(state)
            policies.append(policy)
            values.append(reward)
        
        logger.info(f"Game {game_num} completed after {move_count} moves with result: {result}")
        return states, policies, values

    def get_mcts_policy(self, state) -> np.ndarray:
        """Get policy from MCTS search using valid moves from the game."""
        valid_moves = self.game.get_valid_moves()  # Get valid moves from the game instance
        # Create a policy array of zeros
        policy = np.zeros(7)  # Connect Four has 7 columns
        # Set equal probability for valid moves
        if len(valid_moves) > 0:
            policy[valid_moves] = 1.0 / len(valid_moves)
        return policy

    def generate_training_data(self, num_episodes: int) -> List[Tuple]:
        """Generate training data through self-play."""
        training_data = []
        
        for game_num in range(num_episodes):
            states, policies, values = self.execute_episode(game_num)
            training_data.extend(zip(states, policies, values))
        
        logger.info(f"Total training examples generated: {len(training_data)}")
        return training_data