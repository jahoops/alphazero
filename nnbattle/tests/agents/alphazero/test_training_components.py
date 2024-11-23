import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from torch.utils.data import DataLoader

from nnbattle.agents.alphazero.data_module import ConnectFourDataset, ConnectFourDataModule
from nnbattle.agents.alphazero.lightning_module import ConnectFourLightningModule
from nnbattle.agents.alphazero.train.trainer import train_alphazero

class TestDatasetOperations(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            (np.zeros((2, 6, 7)), np.zeros(7), 0),
            (np.ones((2, 6, 7)), np.ones(7), 1)
        ]
        self.dataset = ConnectFourDataset(self.sample_data)

    def test_dataset_length(self):
        """Test dataset length calculation."""
        self.assertEqual(len(self.dataset), 2)

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        item = self.dataset[0]
        self.assertEqual(len(item), 3)
        self.assertEqual(item[0].shape, torch.Size([2, 6, 7]))
        self.assertEqual(item[1].shape, torch.Size([7]))
        self.assertEqual(item[2].shape, torch.Size([]))

class TestDataModule(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.memory = []
        self.data_module = ConnectFourDataModule(self.agent, num_games=1)

    def test_generate_self_play_games(self):
        """Test self-play games generation."""
        self.agent.self_play = MagicMock()
        self.data_module.generate_self_play_games()
        self.agent.self_play.assert_called_once()

class TestTrainingStep(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.model = MagicMock()
        self.lightning_module = ConnectFourLightningModule(self.agent)

    def test_training_step_with_empty_batch(self):
        """Test training step with empty batch."""
        batch = (
            torch.zeros((0, 2, 6, 7)),  # empty states
            torch.zeros((0, 7)),        # empty mcts_probs
            torch.zeros(0)              # empty rewards
        )
        with self.assertRaises(ValueError):
            self.lightning_module.training_step(batch, 0)

    def test_training_step_with_valid_batch(self):
        """Test training step with valid batch."""
        # Mock trainer reference to avoid warning
        self.lightning_module.trainer = MagicMock()
        
        batch = (
            torch.randn(4, 2, 6, 7),  # 4 samples
            torch.randn(4, 7),
            torch.randn(4)
        )
        self.agent.model.return_value = (
            torch.randn(4, 7, requires_grad=True),   # Ensure tensors have requires_grad=True
            torch.randn(4, 1, requires_grad=True)    # Ensure tensors have requires_grad=True
        )
        
        loss = self.lightning_module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.requires_grad)

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.model = MagicMock()
        self.lightning_module = ConnectFourLightningModule(self.agent)

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        # Mock model parameters
        mock_params = [torch.nn.Parameter(torch.randn(2, 2))]
        self.agent.model.parameters.return_value = mock_params
        
        optimizer = self.lightning_module.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults['lr'], 1e-3)

    def test_configure_optimizers_no_parameters(self):
        """Test optimizer configuration with no parameters."""
        self.agent.model.parameters.return_value = []
        with self.assertRaises(ValueError):
            self.lightning_module.configure_optimizers()

class TestTrainingComponents(unittest.TestCase):
    @patch('nnbattle.agents.alphazero.train.trainer.save_agent_model')
    @patch('nnbattle.agents.alphazero.train.trainer.pl.Trainer')
    @patch('nnbattle.agents.alphazero.train.trainer.ConnectFourLightningModule')
    @patch('nnbattle.agents.alphazero.train.trainer.ConnectFourDataModule')
    @patch('nnbattle.agents.alphazero.train.trainer.initialize_agent')
    def test_train_alphazero_flow(
        self, mock_initialize_agent, mock_data_module, mock_lightning_module, mock_trainer, mock_save_model
    ):
        """Test the overall training flow in train_alphazero."""
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent
        train_alphazero(time_limit=1, num_self_play_games=1, use_gpu=False, load_model=False)
        mock_initialize_agent.assert_called_once()
        mock_data_module.assert_called_once_with(mock_agent, 1)
        mock_lightning_module.assert_called_once_with(mock_agent)
        mock_trainer.return_value.fit.assert_called_once()
        mock_save_model.assert_called()

if __name__ == '__main__':
    unittest.main()