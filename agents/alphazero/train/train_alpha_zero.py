# agents/alphazero/train/train_alpha_zero.py
import os
import time
from datetime import timedelta

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from nnbattle.agents.alphazero import AlphaZeroAgent, Connect4LightningModule, ConnectFourDataModule
from nnbattle.game import ConnectFourGame


def self_play(agent, num_games):
    memory = []
    game = ConnectFourGame()
    for _ in range(num_games):
        game.reset()
        while not game.is_terminal():
            move = agent.select_move(game)
            game.make_move(move)
        result = game.get_result()
        memory.append((game.get_state(), result))
    return memory


def train_alphazero(time_limit, num_self_play_games=100, load_model=True, model_path="nnbattle/agents/alphazero/model/alphazero_model_final.pth"):
    start_time = time.time()
    
    # Initialize Data Module
    data_module = ConnectFourDataModule(memory_size=num_self_play_games)
    
    # Initialize Agent
    agent = AlphaZeroAgent()
    
    # Initialize Lightning Module
    model = Connect4LightningModule(agent=agent)
    
    # Load existing model if required
    if load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    # Set up Model Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='alphazero/checkpoints/',
        filename='alphazero-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_time=timedelta(hours=time_limit),
        callbacks=[checkpoint_callback],
        gpus=1 if torch.cuda.is_available() else 0,
    )
    
    # Start Training
    trainer.fit(model, datamodule=data_module)
    
    # Save the final model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {timedelta(seconds=elapsed_time)}")


if __name__ == "__main__":
    train_alphazero(time_limit=2)  # Example: train for 2 hours