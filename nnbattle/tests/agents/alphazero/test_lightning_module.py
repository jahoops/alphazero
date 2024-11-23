import unittest
import torch
import pytorch_lightning as pl
from unittest.mock import patch
from nnbattle.agents.alphazero.lightning_module import ConnectFourLightningModule
from nnbattle.agents.alphazero.agent_code import initialize_agent

class TestLightningModule(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_agent(load_model=False)
        self.lightning_module = ConnectFourLightningModule(self.agent)

    def test_configure_optimizers(self):
        """Test that the optimizer is correctly configured."""
        optimizer = self.lightning_module.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)

    @patch.object(ConnectFourLightningModule, 'log')  # Mock the 'log' method
    def test_training_step_loss(self, mock_log):
        """Test that the training step returns a valid loss."""
        batch = (
            torch.randn(4, 2, 6, 7),
            torch.softmax(torch.randn(4, self.agent.action_dim), dim=1),
            torch.randn(4)
        )
        trainer = pl.Trainer(fast_dev_run=True, logger=False)  # Disable logger if not needed
        trainer.fit_loop.run_training_epoch = lambda: None  # Prevent actual training
        self.lightning_module.trainer = trainer  # Attach the trainer manually
        loss = self.lightning_module.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        mock_log.assert_called()  # Ensure 'log' was called

    def test_loss_function(self):
        """Test the custom loss function."""
        outputs = (
            torch.randn(4, self.agent.action_dim),
            torch.randn(4, 1)
        )
        targets_policy = torch.softmax(torch.randn(4, self.agent.action_dim), dim=1)
        targets_value = torch.randn(4)
        loss = self.lightning_module.loss_function(outputs, targets_policy, targets_value)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main()