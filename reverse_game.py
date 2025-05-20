# reverse_game.py
"""
Reverse Blackjack environment where dealer plays first.
Inherits from BlackjackEnv but modifies the order of play.
"""

from game import (
    BlackjackEnv, Hand, RESULT_BLACKJACK, RESULT_WIN,
    RESULT_LOSS, RESULT_PUSH, ACTION_HIT, ACTION_STAND,
    ACTION_DOUBLE, ACTION_SPLIT
)

class ReverseBlackjackEnv(BlackjackEnv):
    """
    A variant of Blackjack where the dealer plays first and reveals their
    entire hand before the player acts. This changes the strategy dynamics
    since the player has perfect information about the dealer's hand.
    """
    def reset(self):
        """
        Reset and deal initial hands, but also play out dealer's hand
        before returning first observation.
        """
        # Initialize hands like normal
        self.dealer = Hand()
        self.player_hands = [Hand()]
        self.bets = [1.0]
        self.current_hand = 0
        self.done = False
        self.outcomes = []

        # Deal initial hands
        for _ in range(2):
            self.player_hands[0].add(self.shoe.draw())
            self.dealer.add(self.shoe.draw())

        # Play out dealer's hand immediately
        self._dealer_play_initial()

        return self._get_obs()

    def _dealer_play_initial(self):
        """
        Dealer plays their hand according to rules, but before player acts.
        Does not compute rewards yet since player hasn't acted.
        """
        while True:
            val = self.dealer.value()
            soft = self.dealer.usable_ace()
            if val < 17 or (self.dealer_hits_soft17 and val == 17 and soft):
                self.dealer.add(self.shoe.draw())
            else:
                break

    def _get_obs(self):
        """
        Modified observation that includes dealer's complete hand
        since in this variant we see it from the start.
        """
        hand = self.player_hands[self.current_hand]
        return {
            'player': tuple(hand.cards),
            'dealer_cards': tuple(self.dealer.cards),  # Full dealer hand
            'dealer_value': self.dealer.value(),       # Dealer's final value
            'usable_ace': hand.usable_ace(),
            'can_split': hand.can_split(),
            'can_double': len(hand.cards) == 2
        }

    def _dealer_play(self, info):
        """
        Modified to skip dealer play (already done) and just compute rewards.
        """
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