# play.py
"""
Play one episode using a trained Q-learning agent if available in `saves/`,
otherwise fallback to a RandomAgent. Prints final hands and outcomes.
"""
import os
from game import (
    BlackjackEnv,
    ACTION_HIT,
    ACTION_STAND,
    VALUE_MAP
)
from agents.q_learning import QLearningAgent
from agents.random_agent import RandomAgent
from reverse_game import ReverseBlackjackEnv

if __name__ == "__main__":
    env = ReverseBlackjackEnv()
    agent = QLearningAgent()
    # Try to load trained Q-table
    qpath = 'saves/q_learning.pkl'
    if os.path.exists(qpath):
        agent.load(qpath)
        print(f"Loaded trained Q-learning agent from {qpath}.")
    else:
        agent = RandomAgent()
        print("No trained Q-table found; using RandomAgent.")

    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

    # Print results
    print("Dealer hand:", env.dealer.cards, "value=", env.dealer.value())
    for idx, hand in enumerate(env.player_hands):
        print(f"Player hand {idx+1}:", hand.cards, "value=", hand.value(), "->", env.outcomes[idx])
    print("Total reward:", reward)