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
        current_player = i % 2  # Alternate starting player
        while not game.is_terminal_node():
            agent = agents[current_player]
            move = agent.select_move(game.get_board_state())
            if move is not None and game.is_valid_location(move):
                row = game.get_next_open_row(move)
                game.drop_piece(row, move, game.PLAYER_PIECE if current_player == 0 else game.AI_PIECE)
                if game.winning_move(game.PLAYER_PIECE if current_player == 0 else game.AI_PIECE):
                    results[agent.__class__.__name__] += 1
                    break
                current_player = 1 - current_player
            else:
                # Invalid move or no move available
                results['draws'] += 1
                break
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