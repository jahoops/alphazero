# FILE: tournament/run_tournament.py

import json
import os

from nnbattle.agents.alphazero import AlphaZeroAgent

from nnbattle.agents.minimax import MinimaxAgent
from nnbattle.game.connect_four_game import ConnectFourGame 


def run_tournament(agents, num_games=100):
    results = {agent.__class__.__name__: 0 for agent in agents}
    results['draws'] = 0
    game = ConnectFourGame()

    for i in range(num_games):
        game.reset()
        start_team = (i % 2) + 1  # Alternate starting player
        while not game.is_terminal_node():
            agent = agents[start_team]
            move = agent.select_move(game.get_board_state(), agent.team)
            result = get_game_state()
            if result != "Ongoing":
                if agent.team == result:
                    results[agent.__class__.__name__] += 1
                    break
                elif result == "Draw":
                    results['draws'] += 1
                    break
                else:
                    results[agents[3 - agent.team].__class__.__name__] += 1
        else:
            results['draws'] += 1

    # Save results
    results_dir = os.path.join('tournament', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'tournament_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Tournament completed. Results saved to tournament/results/tournament_results.json")

if __name__ == "__main__":
    agent1 = MinimaxAgent(depth=4)
    agent2 = AlphaZeroAgent()
    agents = [agent1, agent2]
    run_tournament(agents, num_games=100)