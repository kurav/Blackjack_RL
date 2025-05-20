# train_reverse.py
"""
Train a Q-learning agent on Reverse Blackjack.
Saves checkpoints and final Q-table to `saves/reverse/` directory.
"""
from train import train
from reverse_game import ReverseBlackjackEnv
from agents.q_learning import QLearningAgent

if __name__ == "__main__":
    # Hyperparameters
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.2)
    env = ReverseBlackjackEnv()
    train(agent, env, save_dir="saves/reverse") 