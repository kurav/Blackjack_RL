# blackjack/agents/q_learning.py
import random
import pickle
from collections import defaultdict
from typing import Dict, Any, Tuple
from .base import Agent
from game import ACTION_HIT, ACTION_STAND, VALUE_MAP

class QLearningAgent(Agent):
    """
    Tabular Q-learning agent with state discretization by (player_sum, dealer_val, usable_ace).
    Only HIT and STAND are considered in the action space for simplicity.
    Works for both regular and reverse Blackjack.
    """
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # initialize Q for hit and stand only
        self.Q = defaultdict(lambda: {a: 0.0 for a in [ACTION_HIT, ACTION_STAND]})

    def _state_key(self, obs: Dict[str, Any]) -> Tuple[int, int, bool]:
        """
        Map observation to a discrete state:
        - player_sum: total value of player's hand (soft logic applied)
        - dealer_val: numeric value of dealer's upcard (regular) or final value (reverse)
        - usable_ace: whether player has a usable ace
        """
        # compute player's sum with soft‑ace logic
        psum = sum(VALUE_MAP[c] for c in obs['player'])
        if obs['usable_ace'] and psum + 10 <= 21:
            psum += 10

        # Get dealer value (works for both regular and reverse)
        if 'dealer_up' in obs:  # Regular Blackjack
            dealer_val = VALUE_MAP[obs['dealer_up']]
        else:  # Reverse Blackjack
            dealer_val = obs['dealer_value']

        return (psum, dealer_val, obs['usable_ace'])

    def act(self, obs: Dict[str, Any]) -> int:
        state = self._state_key(obs)
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice([ACTION_HIT, ACTION_STAND])
        qvals = self.Q[state]
        return max(qvals, key=qvals.get)

    def observe(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        next_obs: Dict[str, Any],
        done: bool
    ):
        state = self._state_key(obs)
        # compute target
        if done:
            target = reward
        else:
            next_state = self._state_key(next_obs)
            target = reward + self.gamma * max(self.Q[next_state].values())
        # Q-update
        old_val = self.Q[state][action]
        self.Q[state][action] = old_val + self.alpha * (target - old_val)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: {a: 0.0 for a in [ACTION_HIT, ACTION_STAND]}, data)