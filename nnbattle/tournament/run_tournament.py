# FILE: tournament/run_tournament.py

import json
import os
import logging

from nnbattle.agents.alphazero import AlphaZeroAgent
from nnbattle.agents.minimax.agent_code import MinimaxAgent  # Ensure correct import path
from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError
from nnbattle.constants import RED_TEAM, YEL_TEAM

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tournament(agents, num_games=100):
    results = {agent.__class__.__name__: 0 for agent in agents}
    results['draws'] = 0
    game = ConnectFourGame()

    for i in range(num_games):
        game.reset()
        # Set starting team based on game index
        start_team = RED_TEAM if (i % 2) == 0 else YEL_TEAM  # Assign RED_TEAM or YEL_TEAM
        while game.get_game_state() == "ONGOING":
            agent = next((a for a in agents if a.team == start_team), None)
            if agent is None:
                logger.error(f"No agent found for team {start_team}.")
                break
            selected_action = agent.select_move(game)
            game.make_move(selected_action, agent.team)
            result = game.get_game_state()
            if result != "ONGOING":
                if result in [RED_TEAM, YEL_TEAM]:
                    winner_agent = next((a for a in agents if a.team == result), None)
                    if winner_agent:
                        results[winner_agent.__class__.__name__] += 1
                elif result == "Draw":
                    results['draws'] += 1
        # Add board representation after each game
        logger.info(f"Final board for game {i + 1}:\n{game.board_to_string()}")

    # Save results
    results_dir = os.path.join('tournament', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'tournament_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Tournament completed. Results saved to tournament/results/tournament_results.json")

if __name__ == "__main__":
    agent1 = MinimaxAgent(depth=4, team=YEL_TEAM)  # Set MinimaxAgent to YEL_TEAM
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
    run_tournament(agents, num_games=100)