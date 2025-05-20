# blackjack/game.py
"""
Core classes and constants for the Blackjack RL environment.
- ACTION_* constants name the four legal moves.
- VALUE_MAP defines card values (Ace low; soft logic applied in Hand).
- Shoe: manages a shuffled deck (or multiple decks) and reshuffle logic.
- Hand: stores cards, computes hard/soft totals, bust/split eligibility.
- BlackjackEnv: orchestrates deals, player actions, dealer policy, scoring, and logs outcomes including push.
"""
import random
from collections import deque
from typing import List, Tuple, Dict, Any

# --- ACTIONS ---------------------------------------------------------
ACTION_HIT    = 0  # draw another card
ACTION_STAND  = 1  # stop drawing and let dealer play
ACTION_DOUBLE = 2  # double bet, draw exactly one card, then stand
ACTION_SPLIT  = 3  # split a pair into two hands

# --- CARD VALUES -----------------------------------------------------
VALUE_MAP = {
    'A': 1,
    '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 10, 'Q': 10, 'K': 10,
}

# --- RESULTS ---------------------------------------------------------
RESULT_BLACKJACK = 'blackjack'
RESULT_WIN       = 'win'
RESULT_LOSS      = 'loss'
RESULT_PUSH      = 'push'

class Shoe:
    """
    A shoe of one or more shuffled decks.  Automatically reshuffles
    when remaining cards fall below a threshold.
    """
    def __init__(self, num_decks: int = 1, reshuffle_threshold: int = 15):
        self.num_decks = num_decks
        self.reshuffle_threshold = reshuffle_threshold
        self._build_shoe()

    def _build_shoe(self) -> None:
        ranks = ['A'] + [str(n) for n in range(2, 11)] + ['J', 'Q', 'K']
        deck = ranks * 4 * self.num_decks
        random.shuffle(deck)
        self.cards: deque[str] = deque(deck)

    def draw(self) -> str:
        if len(self.cards) < self.reshuffle_threshold:
            self._build_shoe()
        return self.cards.popleft()

class Hand:
    """
    A single blackjack hand: holds cards and provides
    value(), usable_ace(), is_bust(), can_split().
    """
    def __init__(self, cards: List[str] = None):
        self.cards = cards[:] if cards else []

    def add(self, card: str) -> None:
        self.cards.append(card)

    def value(self) -> int:
        total = sum(VALUE_MAP[c] for c in self.cards)
        # treat one Ace as 11 if it doesn't bust
        if 'A' in self.cards and total + 10 <= 21:
            return total + 10
        return total

    def usable_ace(self) -> bool:
        return 'A' in self.cards and self.value() != sum(VALUE_MAP[c] for c in self.cards)

    def is_bust(self) -> bool:
        return self.value() > 21

    def can_split(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

class BlackjackEnv:
    """
    Encapsulates a full round of Blackjack between one player and dealer.
    Logs individual hand outcomes: blackjack, win, loss, or push.

    Methods:
    - reset(): deal initial hands and return the first observation.
    - step(action): advance the game by one player action.
    """
    def __init__(
        self,
        natural_payout: float = 1.5,
        dealer_hits_soft17: bool = False,
        num_decks: int = 1,
        reshuffle_threshold: int = 15
    ):
        self.natural_payout = natural_payout
        self.dealer_hits_soft17 = dealer_hits_soft17
        self.shoe = Shoe(num_decks, reshuffle_threshold)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.dealer = Hand()
        self.player_hands: List[Hand] = [Hand()]
        self.bets: List[float] = [1.0]
        self.current_hand = 0
        self.done = False
        self.outcomes: List[str] = []  # per-hand results
        # initial deal: 2 cards each
        for _ in range(2):
            self.player_hands[0].add(self.shoe.draw())
            self.dealer.add(self.shoe.draw())
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        hand = self.player_hands[self.current_hand]
        return {
            'player': tuple(hand.cards),
            'dealer_up': self.dealer.cards[0],
            'usable_ace': hand.usable_ace(),
            'can_split': hand.can_split(),
            'can_double': len(hand.cards) == 2
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert not self.done, "Episode over; call reset() first."
        hand = self.player_hands[self.current_hand]
        info: Dict = {}

        if action == ACTION_HIT:
            hand.add(self.shoe.draw())
            if hand.is_bust():
                return self._finalize_hand(-self.bets[self.current_hand], info)
            return self._get_obs(), 0.0, False, info

        if action == ACTION_STAND:
            return self._next_or_dealer(info)

        if action == ACTION_DOUBLE:
            if len(hand.cards) != 2:
                raise ValueError("Double only on first two cards")
            self.bets[self.current_hand] *= 2
            hand.add(self.shoe.draw())
            return self._next_or_dealer(info)

        if action == ACTION_SPLIT:
            if not hand.can_split():
                raise ValueError("Split only on identical pairs")
            card = hand.cards.pop()
            new_hand = Hand([card])
            self.player_hands.insert(self.current_hand + 1, new_hand)
            self.bets.insert(self.current_hand + 1, self.bets[self.current_hand])
            hand.add(self.shoe.draw())
            new_hand.add(self.shoe.draw())
            return self._get_obs(), 0.0, False, info

        raise ValueError("Unknown action")

    def _next_or_dealer(self, info: Dict) -> Tuple:
        """Handle end of a hand or proceed to dealer play."""
        hand = self.player_hands[self.current_hand]
        reward = -self.bets[self.current_hand] if hand.is_bust() else None
        self.current_hand += 1
        if self.current_hand < len(self.player_hands):
            return self._get_obs(), reward or 0.0, False, info
        return self._dealer_play(info)

    def _dealer_play(self, info: Dict) -> Tuple:
        """Dealer draws and final payoffs with outcome logging."""
        while True:
            val = self.dealer.value()
            soft = self.dealer.usable_ace()
            if val < 17 or (self.dealer_hits_soft17 and val == 17 and soft):
                self.dealer.add(self.shoe.draw())
            else:
                break

        rewards = []
        for hand, bet in zip(self.player_hands, self.bets):
            hv = hand.value()
            d_val = self.dealer.value()
            d_bust = d_val > 21
            if len(hand.cards) == 2 and hv == 21:
                self.outcomes.append(RESULT_BLACKJACK)
                rewards.append(bet * self.natural_payout)
            elif hv > 21:
                self.outcomes.append(RESULT_LOSS)
                rewards.append(-bet)
            elif d_bust or hv > d_val:
                self.outcomes.append(RESULT_WIN)
                rewards.append(bet)
            elif hv < d_val:
                self.outcomes.append(RESULT_LOSS)
                rewards.append(-bet)
            else:
                self.outcomes.append(RESULT_PUSH)
                rewards.append(0.0)

        total_reward = sum(rewards)
        self.done = True
        return {}, total_reward, True, info

    def _finalize_hand(self, reward: float, info: Dict) -> Tuple:
        """Called when a player hand busts; advances or ends episode."""
        self.outcomes.append(RESULT_LOSS)
        self.current_hand += 1
        if self.current_hand < len(self.player_hands):
            return self._get_obs(), reward, False, info
        self.done = True
        return {}, reward, True, info


