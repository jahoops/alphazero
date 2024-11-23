
import unittest
from nnbattle.agents.alphazero.data_module import ConnectFourDataModule
from nnbattle.agents.alphazero.agent_code import initialize_agent

class TestDataModule(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_agent(load_model=False)
        self.data_module = ConnectFourDataModule(self.agent, num_games=1)

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly."""
        self.assertEqual(len(self.data_module.dataset), 0)

    def test_generate_self_play_games(self):
        """Test that self-play games are generated and data is collected."""
        self.data_module.generate_self_play_games()
        self.assertGreater(len(self.data_module.dataset), 0)
        sample = self.data_module.dataset[0]
        state, mcts_prob, reward = sample
        self.assertEqual(state.shape, (2, 6, 7))
        self.assertEqual(mcts_prob.shape, (self.agent.action_dim,))
        self.assertIsInstance(reward.item(), float)

    def test_train_dataloader_output(self):
        """Test that the DataLoader provides batches with correct shapes."""
        self.data_module.generate_self_play_games()
        dataloader = self.data_module.train_dataloader()
        batch = next(iter(dataloader))
        states, mcts_probs, rewards = batch
        self.assertEqual(states.shape[1:], (2, 6, 7))
        self.assertEqual(mcts_probs.shape[1], self.agent.action_dim)
        self.assertEqual(rewards.ndim, 1)

if __name__ == '__main__':
    unittest.main()