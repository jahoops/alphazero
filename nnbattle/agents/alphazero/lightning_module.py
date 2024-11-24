# /lightning_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nnbattle.agents.alphazero.network import Connect4Net  # Ensure correct import if needed

class ConnectFourLightningModule(pl.LightningModule):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.loss_fn = self.loss_function
        self.save_hyperparameters()
        self.automatic_optimization = True

    def forward(self, x):
        # Ensure x has shape [batch_size, 2, 6, 7]
        assert x.shape[1:] == (2, 6, 7), f"Input tensor has incorrect shape: {x.shape}"
        return self.agent.model(x)

    def training_step(self, batch, batch_idx):
        states, mcts_probs, rewards = batch
        
        # Forward pass
        logits, values = self.forward(states)
        
        # Ensure proper shapes
        values = values.squeeze()
        rewards = rewards.float()
        
        # Calculate losses with added small epsilon to prevent log(0)
        value_loss = F.mse_loss(values, rewards)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
        total_loss = value_loss + policy_loss

        # Log metrics without _step suffix
        self.log('value_loss', value_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('policy_loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss  # Return just the loss tensor

    def on_train_epoch_end(self):
        # Get metrics that were logged during training steps
        metrics = self.trainer.callback_metrics
        
        # Log epoch-level metrics if they exist
        if 'value_loss_step' in metrics:
            self.log('value_loss_epoch', metrics['value_loss_step'], on_epoch=True, prog_bar=True)
        if 'policy_loss_step' in metrics:
            self.log('policy_loss_epoch', metrics['policy_loss_step'], on_epoch=True, prog_bar=True)
        if 'train_loss_step' in metrics:
            self.log('train_loss_epoch', metrics['train_loss_step'], on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }

    def loss_function(self, outputs, targets_policy, targets_value):
        logits, values = outputs
        value_loss = F.mse_loss(values.squeeze(), targets_value)
        policy_loss = -torch.mean(torch.sum(targets_policy * F.log_softmax(logits, dim=1), dim=1))
        return value_loss + policy_loss

# Ensure __all__ is defined for easier imports
__all__ = ['ConnectFourLightningModule']