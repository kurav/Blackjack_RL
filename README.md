# Playing blackjack using RL - does the house always win?

This project implements a Reinforcement Learning environment for both standard and reverse Blackjack, where agents can learn optimal playing strategies using Q-learning.

## 🎮 Game Variants

### Standard Blackjack
- Traditional Blackjack rules where the player acts first
- Dealer reveals one card initially
- Standard payout rules (3:2 for blackjack)

### Reverse Blackjack
- A variant where the dealer plays first
- Player has perfect information about dealer's hand
- Changes strategy dynamics significantly
- Useful for studying perfect information scenarios

## 🏗️ Project Structure

```
blackjack/
├── agents/           # RL agent implementations
├── saves/           # Training checkpoints and Q-tables
├── visualization_results/  # Training visualization outputs
├── game.py          # Core Blackjack environment
├── reverse_game.py  # Reverse Blackjack variant
├── train.py         # Training utilities
├── train_reverse.py # Reverse Blackjack training
├── play.py          # Interactive gameplay
└── visualize.py     # Training visualization tools
```

## 🎯 Usage

### Training an Agent

1. Train a standard Blackjack agent:
  ```bash
  python train.py
  ```

2. Train a Reverse Blackjack agent:
  ```bash
  python train_reverse.py
  ```

Training checkpoints and final Q-tables are saved in the `saves/` directory.

### Playing Against Trained Agents

To play against a trained agent:
```bash
python play.py
```

### Visualizing Training Results

To visualize training progress and results:
```bash
python visualize.py
```

## 🎲 Game Rules

### Standard Blackjack
- Dealer hits on soft 17
- Blackjack pays 3:2
- Double down allowed on any two cards
- Split pairs allowed
- No insurance or surrender

### Reverse Blackjack
- Same rules as standard Blackjack
- Dealer plays first and reveals complete hand
- Player has perfect information
- Strategic decisions are based on known dealer outcome

## 🤖 Reinforcement Learning Details

The project uses Q-learning with the following parameters:
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.99
- Exploration rate (epsilon): 0.2

The state space includes:
- Player's cards
- Dealer's up card (or complete hand in reverse variant)
- Usable ace status
- Split/double down eligibility

## 📊 Results

Training results and visualizations are stored in the `visualization_results/` directory, including:
- Learning curves
- Strategy heatmaps
- Win rate analysis
