# /lightning_module.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent_code import AlphaZeroAgent
from .network import Connect4Net

class Connect4LightningModule(pl.LightningModule):
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        super(Connect4LightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = Connect4Net(state_dim, action_dim)
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, mcts_probs, values = batch
        log_policy, predicted_value = self.forward(states)
        
        # Compute losses
        loss_policy = self.criterion_policy(log_policy, torch.argmax(mcts_probs, dim=1))
        loss_value = self.criterion_value(predicted_value.squeeze(), values)
        loss = loss_policy + loss_value
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer