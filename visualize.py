# visualize.py
"""
Visualize training results for Blackjack.
Creates simple plots of win rates and profit over time.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_q_table(filepath):
    """Load Q-table from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_metrics(log_file):
    """Extract key metrics from training log."""
    metrics = {
        'episodes': [],
        'win_rates': [],
        'profit_per_hand': []
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Episodes' in line:
                # Extract episode number
                ep = int(line.split('–')[1].strip())
                metrics['episodes'].append(ep)
            elif 'Win Rate:' in line:
                rate = float(line.split(':')[1].strip().rstrip('%')) / 100
                metrics['win_rates'].append(rate)
            elif 'Profit per Hand:' in line:
                profit = float(line.split('$')[1].strip())
                metrics['profit_per_hand'].append(profit)
    
    return metrics

def plot_metrics(metrics, save_dir, title):
    """Plot win rates and profit per hand."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)
    
    # Plot win rates
    ax1.plot(metrics['episodes'], metrics['win_rates'], 'b-', label='Win Rate')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Win Rate')
    ax1.grid(True)
    ax1.legend()
    
    # Plot profit per hand
    ax2.plot(metrics['episodes'], metrics['profit_per_hand'], 'g-', label='Profit per Hand')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Profit per Hand ($)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_strategy(q_table, save_dir, title):
    """Plot learned strategy as a heatmap showing hit/stand decisions."""
    # Determine if this is reverse Blackjack by checking dealer values
    is_reverse = any(dealer_val > 11 for _, dealer_val, _ in q_table.keys())
    
    if is_reverse:
        dealer_values = range(17, 23)  # 17-22 (including bust)
        dealer_labels = [str(v) for v in range(17, 22)] + ['Bust']
    else:
        dealer_values = range(2, 12)   # 2-11
        dealer_labels = [str(v) for v in range(2, 12)]

    # Create matrices for hard and soft hands
    hard_hands = range(5, 22)  # 5-21
    soft_hands = range(13, 22)  # 13-21
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Learned Strategy - {title}', fontsize=14)
    
    # Initialize matrices
    hard_matrix = np.zeros((len(hard_hands), len(dealer_values)))
    soft_matrix = np.zeros((len(soft_hands), len(dealer_values)))
    
    # Fill matrices based on Q-values
    for dealer_val in dealer_values:
        dealer_idx = dealer_val - (17 if is_reverse else 2)
        
        # Process hard hands
        for player_sum in hard_hands:
            state = (player_sum, dealer_val, False)
            if state in q_table:
                action = max(q_table[state].items(), key=lambda x: x[1])[0]
                hard_matrix[player_sum-5, dealer_idx] = action
        
        # Process soft hands
        for player_sum in soft_hands:
            state = (player_sum, dealer_val, True)
            if state in q_table:
                action = max(q_table[state].items(), key=lambda x: x[1])[0]
                soft_matrix[player_sum-13, dealer_idx] = action
    
    # Plot hard hands
    im1 = ax1.imshow(hard_matrix, cmap='RdYlBu', aspect='auto')
    ax1.set_title('Hard Hands (No Ace)')
    ax1.set_xlabel('Dealer Value')
    ax1.set_ylabel('Player Total')
    ax1.set_xticks(range(len(dealer_labels)))
    ax1.set_xticklabels(dealer_labels)
    ax1.set_yticks(range(len(hard_hands)))
    ax1.set_yticklabels(hard_hands)
    ax1.grid(True)
    
    # Add colorbar for hard hands
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_ticks([0, 1])
    cbar1.set_ticklabels(['Hit', 'Stand'])
    
    # Plot soft hands
    im2 = ax2.imshow(soft_matrix, cmap='RdYlBu', aspect='auto')
    ax2.set_title('Soft Hands (With Ace)')
    ax2.set_xlabel('Dealer Value')
    ax2.set_ylabel('Player Total')
    ax2.set_xticks(range(len(dealer_labels)))
    ax2.set_xticklabels(dealer_labels)
    ax2.set_yticks(range(len(soft_hands)))
    ax2.set_yticklabels(soft_hands)
    ax2.grid(True)
    
    # Add colorbar for soft hands
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(['Hit', 'Stand'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'strategy_{title.lower().replace(" ", "_")}.png'))
    plt.close()

def main():
    # Create output directory
    save_dir = 'visualization_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Process regular Blackjack results
    regular_metrics = extract_metrics('saves/training.log')
    regular_q_table = load_q_table('saves/q_learning.pkl')
    
    # Process reverse Blackjack results
    reverse_metrics = extract_metrics('saves/reverse/training.log')
    reverse_q_table = load_q_table('saves/reverse/q_learning.pkl')
    
    # Plot metrics
    plot_metrics(regular_metrics, save_dir, 'Regular Blackjack')
    plot_metrics(reverse_metrics, save_dir, 'Reverse Blackjack')
    
    # Plot strategies
    plot_strategy(regular_q_table, save_dir, 'Regular Blackjack')
    plot_strategy(reverse_q_table, save_dir, 'Reverse Blackjack')

if __name__ == "__main__":
    main()