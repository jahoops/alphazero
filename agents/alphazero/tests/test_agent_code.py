# /tests/test_agent_code.py

import unittest
from unittest.mock import MagicMock, patch

from alphazero.agent_code import AlphaZeroAgent
from game.connect_four_game import \
    ConnectFourGame  # Update the import path as needed


class TestAlphaZeroAgent(unittest.TestCase):
    @patch('.agent_code.Connect4Net')
    @patch('.agent_code.MCTSNode')
    def setUp(self, mock_mcts_node, mock_connect4net):
        self.agent = AlphaZeroAgent(
            state_dim=42,
            action_dim=7,
            use_gpu=False,
            model_path="/model/test_model.pth"
        )
        # Mock the neural network and MCTSNode
        self.agent.model = mock_connect4net.return_value
        self.agent.model.eval = MagicMock()
        mock_mcts_node.return_value = MagicMock()

    @patch('.agent_code.torch.load')
    def test_load_model_success(self, mock_torch_load):
        # Simulate successful model loading
        mock_torch_load.return_value = {}
        self.agent.load_model()
        self.assertTrue(self.agent.model_loaded)
        self.agent.model.load_state_dict.assert_called()

    @patch('.agent_code.torch.load')
    def test_load_model_failure(self, mock_torch_load):
        # Simulate model loading failure
        mock_torch_load.side_effect = Exception("Load failed")
        with self.assertRaises(Exception):
            self.agent.load_model()
        self.assertFalse(self.agent.model_loaded)

    def test_select_move_no_model_loaded(self):
        # Ensure model is loaded if not already
        with patch.object(self.agent, 'load_model') as mock_load_model:
            self.agent.model_loaded = False
            self.agent.mcts_simulate = MagicMock(return_value=(3, [3,4], [0.6, 0.4]))
            action = self.agent.select_move(ConnectFourGame())
            mock_load_model.assert_called_once()
            self.assertEqual(action, 3)

    def test_select_move_model_already_loaded(self):
        # Ensure model is not loaded again if already loaded
        with patch.object(self.agent, 'load_model') as mock_load_model:
            self.agent.model_loaded = True
            self.agent.mcts_simulate = MagicMock(return_value=(4, [2,5], [0.7, 0.3]))
            action = self.agent.select_move(ConnectFourGame())
            mock_load_model.assert_not_called()
            self.assertEqual(action, 4)

    def test_select_move_no_actions(self):
        # Simulate scenario where no actions are available
        self.agent.mcts_simulate = MagicMock(return_value=(None, [], []))
        action = self.agent.select_move(ConnectFourGame())
        self.assertIsNone(action)

    def test_self_play_deterministic(self):
        # Test self-play game data storage
        with patch.object(self.agent, 'select_move', return_value=3):
            with patch.object(self.agent.env, 'step', return_value=(np.zeros((6,7), dtype=np.int8), 0, True, {})):
                self.agent.self_play()
                self.assertEqual(len(self.agent.memory), 1)
                state, mcts_prob, value = self.agent.memory[0]
                self.assertEqual(mcts_prob[3], 1.0)
                self.assertEqual(value, -1)  # Assuming player 1 lost

if __name__ == '__main__':
    unittest.main()