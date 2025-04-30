# Deep Q-Network (DQN)  
**Type**: *Value-Based, Off-Policy*  

## What It Is  
DQN combines Q-learning with deep neural networks to handle complex environments (e.g., video games). Instead of a Q-table, it uses a neural network to approximate Q-values.

## How It Works  
- **Experience Replay**: Stores past experiences (state, action, reward, next state) and randomly samples them during training to break correlations.  
- **Target Network**: A separate network to stabilize learning (prevents feedback loops).  
- **Epsilon-Greedy Strategy**: Balances exploration (random actions) and exploitation (best-known actions).  

## When to Use It  
- High-dimensional inputs (e.g., pixels from a game screen).  
- Discrete action spaces (e.g., joystick directions).  

## Key Ideas  
- **Deep Learning + Q-Learning**: Neural network replaces the Q-table.  
- **Stable Training**: Experience replay and target networks prevent divergence.  