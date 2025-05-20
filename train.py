"""
Train a Q-learning agent on Blackjack.
Saves checkpoints and final Q-table to `saves/` directory.
"""
import os
import time
from game import BlackjackEnv
from agents.q_learning import QLearningAgent

def train(
    agent,
    env,
    num_episodes: int = 500000,
    save_every: int = 10000,
    save_dir: str = "saves",
    initial_money: float = 10000,
    bet_size: float = 100
):
    """
    Train agent for specified number of episodes.
    Saves checkpoints every save_every episodes.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Open log file with UTF-8 encoding
    log_file = os.path.join(save_dir, "training.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Starting training for {num_episodes} episodes...\n")
        start_time = time.time()
        
        # Window counters (reset every save_every)
        window_wins = window_losses = window_pushes = window_bj = 0
        window_money = initial_money
        window_start_money = initial_money

        for ep in range(1, num_episodes + 1):
            # Run one episode
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(obs, action, reward if done else 0.0, next_obs, done)
                obs = next_obs
                total_reward += reward

            # Track outcomes
            for outcome in env.outcomes:
                if outcome == "win":
                    window_wins += 1
                    window_money += bet_size
                elif outcome == "loss":
                    window_losses += 1
                    window_money -= bet_size
                elif outcome == "push":
                    window_pushes += 1
                elif outcome == "blackjack":
                    window_bj += 1
                    window_money += bet_size * 1.5

            # Every save_every episodes, log and save
            if ep % save_every == 0:
                # Calculate window statistics
                window_total = window_wins + window_losses + window_pushes + window_bj
                window_profit = window_money - window_start_money
                profit_per_hand = window_profit / window_total if window_total > 0 else 0

                # Print to console
                print(f"\nEpisodes {ep - save_every + 1:>6}–{ep:<6}")
                print(f"Win Rate: {window_wins/window_total:.1%}")
                print(f"Loss Rate: {window_losses/window_total:.1%}")
                print(f"Push Rate: {window_pushes/window_total:.1%}")
                print(f"Blackjack Rate: {window_bj/window_total:.1%}")
                print(f"Profit per Hand: ${profit_per_hand:.2f}")
                print(f"Window Profit: ${window_profit:,.2f}")
                print(f"Current Balance: ${window_money:,.2f}")

                # Write to log file
                f.write(f"\nEpisodes {ep - save_every + 1:>6}–{ep:<6}\n")
                f.write(f"Win Rate: {window_wins/window_total:.1%}\n")
                f.write(f"Loss Rate: {window_losses/window_total:.1%}\n")
                f.write(f"Push Rate: {window_pushes/window_total:.1%}\n")
                f.write(f"Blackjack Rate: {window_bj/window_total:.1%}\n")
                f.write(f"Profit per Hand: ${profit_per_hand:.2f}\n")
                f.write(f"Window Profit: ${window_profit:,.2f}\n")
                f.write(f"Current Balance: ${window_money:,.2f}\n")
                f.flush()

                # Save checkpoint
                agent.save(os.path.join(save_dir, f"q_learning_{ep}.pkl"))

                # Reset window counters
                window_wins = window_losses = window_pushes = window_bj = 0
                window_start_money = window_money

        # Save final Q-table
        agent.save(os.path.join(save_dir, "q_learning.pkl"))
        elapsed = time.time() - start_time
        f.write(f"\nTraining complete in {elapsed:.1f} seconds\n")
        f.write(f"Final Q-table saved to {save_dir}/q_learning.pkl\n")

if __name__ == "__main__":
    # Hyperparameters
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.2)
    #agent = SARSAgent(alpha=0.1,gamma=0.99,epsilon=0.2)
    env = BlackjackEnv()
    train(agent, env)