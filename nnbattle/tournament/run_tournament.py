# FILE: tournament/run_tournament.py

import json
import os
import logging

from nnbattle.agents.alphazero import AlphaZeroAgent
from nnbattle.agents.minimax.agent_code import MinimaxAgent  # Ensure correct import path
from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from nnbattle.constants import RED_TEAM, YEL_TEAM

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tournament(agents, num_games=10):
    # Format agent names with team strings instead of numbers
    results = {f"{agent.__class__.__name__} ({'RED_TEAM' if agent.team == RED_TEAM else 'YEL_TEAM'})": 0 for agent in agents}
    results['draws'] = 0
    game = ConnectFourGame()

    for i in range(num_games):
        game.reset()
        current_team = RED_TEAM if i % 2 == 0 else YEL_TEAM  # Alternate starting team
        logger.info(f"Starting Game {i+1}: Team {current_team} starts")

        while game.get_game_state() == "ONGOING":
            agent = next((a for a in agents if a.team == current_team), None)
            if agent is None:
                logger.error(f"No agent found for team {current_team}")
                break
                
            # Handle different return types from select_move
            move_result = agent.select_move(game)
            selected_action = move_result[0] if isinstance(move_result, tuple) else move_result
            logger.debug(f"Agent {agent.__class__.__name__} ({agent.team}) selects column {selected_action}")
            
            try:
                move_successful = game.make_move(selected_action, agent.team)
                if move_successful:
                    logger.info(f"Team {agent.team} placed piece in column {selected_action}")
                    logger.info(f"Board state:\n{game.board_to_string()}")
                else:
                    logger.error(f"Move unsuccessful for team {agent.team} in column {selected_action}")
                    break
            except (InvalidMoveError, InvalidTurnError) as e:
                logger.error(f"Invalid move by {agent.__class__.__name__}: {e}")
                break

            current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM

        # Log game result
        result = game.get_game_state()
        logger.info(f"Game {i+1} ended with result: {result}")
        logger.info(f"Final board:\n{game.board_to_string()}")
        
        if result in [RED_TEAM, YEL_TEAM]:
            winner = next((a for a in agents if a.team == result), None)
            if winner:
                team_str = 'RED_TEAM' if result == RED_TEAM else 'YEL_TEAM'
                results[f"{winner.__class__.__name__} ({team_str})"] += 1
        else:
            results['draws'] += 1

    # Save results
    results_dir = os.path.join('tournament', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'tournament_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Tournament completed. Results saved to tournament/results/tournament_results.json")

if __name__ == "__main__":
    agent1 = MinimaxAgent(depth=3, team=YEL_TEAM)  # Set MinimaxAgent to YEL_TEAM
    agent2 = AlphaZeroAgent(
        action_dim=7,
        state_dim=2,
        use_gpu=True,
        num_simulations=800,
        c_puct=1.4,
        load_model=True,
        team=RED_TEAM  # Set AlphaZeroAgent to RED_TEAM
    )
    agents = [agent1, agent2]
    run_tournament(agents, num_games=10)