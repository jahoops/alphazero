# /lightning_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ConnectFourLightningModule(pl.LightningModule):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.loss_fn = self.loss_function

    def forward(self, x):
        return self.agent.model(x)

    def training_step(self, batch, batch_idx):
        states, mcts_probs, rewards = batch
        logits, values = self.forward(states)
        value_loss = F.mse_loss(values.squeeze(), rewards)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits, dim=1), dim=1))
        loss = value_loss + policy_loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_function(self, outputs, targets_policy, targets_value):
        logits, values = outputs
        value_loss = F.mse_loss(values.squeeze(), targets_value)
        policy_loss = -torch.mean(torch.sum(targets_policy * F.log_softmax(logits, dim=1), dim=1))
        return value_loss + policy_loss