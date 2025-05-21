# blackjack/agents/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

class Agent(ABC):
    """
    Base class for Blackjack agents.
    Agents must implement:
      - act(observation) -> action
      - observe(obs, action, reward, next_obs, done)
    """
    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def observe(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        next_obs: Dict[str, Any],
        done: bool
    ) -> None:
        pass