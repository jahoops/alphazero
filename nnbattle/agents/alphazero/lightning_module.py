# /lightning_module.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .network import Connect4Net


class Connect4LightningModule(pl.LightningModule):
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        super().__init__()
        self.model = Connect4Net(state_dim, action_dim)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, mcts_probs, values = batch
        predicted_values = self.model(states)
        loss = self.loss_function(predicted_values, values)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def loss_function(self, predicted_values, target_values):
        return nn.MSELoss()(predicted_values, target_values)