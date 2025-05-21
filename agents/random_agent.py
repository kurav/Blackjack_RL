# blackjack/agents/random_agent.py
import random
from .base import Agent
from game import ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT

class RandomAgent(Agent):
    """Chooses uniformly at random among legal actions."""
    def __init__(self):
        self.actions = [ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT]

    def act(self, obs):
        # filter illegal: double only on 2 cards, split only when allowed
        legal = []
        for a in self.actions:
            if a == ACTION_DOUBLE and not obs['can_double']:
                continue
            if a == ACTION_SPLIT and not obs['can_split']:
                continue
            legal.append(a)
        return random.choice(legal)

    def observe(self, obs, action, reward, next_obs, done):
        # random agent learns nothing
        return