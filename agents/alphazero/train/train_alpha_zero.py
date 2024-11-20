# /train/train_alpha_zero.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from alphazero.lightning_module import Connect4LightningModule
from alphazero.agent_code import AlphaZeroAgent
from game.connect_four_game import ConnectFourGame
import torch
import time
import numpy as np

def self_play(agent, num_games):
    memory = []
    for _ in range(num_games):
        game = ConnectFourGame()
        states, mcts_probs, values = [], [], []
        while not game.is_terminal_node():
            state = game.get_board_state()
            move, mcts_prob = agent.select_move(state)
            states.append(state)
            mcts_probs.append(mcts_prob)
            game.make_move(move)
            if game.is_terminal_node():
                value = game.get_reward()
                values.extend([value] * len(states))
                memory.extend(zip(states, mcts_probs, values))
                break
    return memory

def train_alphazero(time_limit, num_self_play_games=100, load_model=True, model_path="/model/alphazero_model_final.pth"):
    # Initialize the agent
    state_dim = 42  # Example: 6 rows * 7 columns
    action_dim = 7  # Number of columns in Connect Four
    agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=True, model_path=model_path)
    
    # Load existing model if required
    if load_model:
        agent.load_model()
    
    # Initialize the LightningModule
    lightning_model = Connect4LightningModule(state_dim=state_dim, action_dim=action_dim, learning_rate=0.001)
    
    # Initialize Trainer with checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='/checkpoints/',
        filename='model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_time={"hours": time_limit},
        gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
        callbacks=[checkpoint_callback]
    )
    
    start_time = time.time()
    while time.time() - start_time < time_limit * 3600:
        # Self-play to generate training data
        memory = self_play(agent, num_self_play_games)
        
        # Update agent's memory
        agent.memory = memory
        
        # Train the model
        trainer.fit(lightning_model)
        
        # Save the model
        agent.save_model(model_path)

if __name__ == "__main__":
    train_alphazero(time_limit=2)  # Example: train for 2 hours