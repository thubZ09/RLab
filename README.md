# Reinforcement Learning Algorithms (PyTorch)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of modern RL algorithms implemented in PyTorch, designed for clarity and educational purposes. Each implementation includes mathematical formulations, testing suites, and practical examples.

### **Implemented Algorithms**:
  - Q-Learning (Tabular)
  - SARSA (Tabular)
  - Deep Q-Network (DQN)
  - REINFORCE (Policy Gradient)
  - Actor-Critic (A2C)
  - Proximal Policy Optimization (PPO)
  - Direct Preference Optimization (DPO)
  - Trust Region Policy Optimization (TRPO)

### **Key Components**:
  - Modular agent architectures
  - Custom environments (GridWorld, Continuous Control)
  - Comprehensive test suite
  - Jupyter notebook examples
  - Hyperparameter configurations
  - PyTorch-optimized implementations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RLab.git
cd RLab
```

2. Install dependencies:  
```
pip install -r requirements.txt
```
### Requirements

- Python 3.8+

- PyTorch 2.0+

- Gymnasium 0.28+

- NumPy

- Matplotlib

## Usage
### Basic Q-Learning Example
```
from agents.q_learning import QLearningAgent
from environments.gridworld import GridWorld

env = GridWorld()
agent = QLearningAgent(num_states=25, num_actions=4)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
```

### Training PPO on CartPole
```
from agents.ppo import PPOAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = PPOAgent(state_dim=4, action_dim=2)

# See examples/cartpole_ppo.ipynb for full training loop
```
## Key Concepts
### Algorithm Selection Guide

| Algorithm       | Best For                          | Action Space   | Training Stability | Key Features                         |
|-----------------|-----------------------------------|----------------|--------------------|--------------------------------------|
| **Q-Learning**  | Discrete state/action problems    | Discrete       | Medium             | Tabular, off-policy, simple          |
| **SARSA**       | Risk-sensitive environments       | Discrete       | Medium             | On-policy, conservative updates      |
| **DQN**         | High-dimensional observations     | Discrete       | Medium             | Experience replay, target networks   |
| **REINFORCE**   | Episodic tasks, policy gradients  | Discrete/Cont. | Low                | Monte Carlo updates, high variance    |
| **Actor-Critic**| Continuous control tasks          | Continuous     | Medium             | Low variance, combined value/policy  |
| **PPO**         | Complex continuous control        | Continuous     | High               | Clipped objectives, stable updates   |
| **DPO**         | Human preference alignment        | Both           | High               | Direct optimization, no reward model |
| **TRPO**        | Safe policy updates               | Continuous     | Very High          | KL constraints, guaranteed monotonic |

## Contributing  
Contributions are welcome! Please follow these steps:

- Fork the repository

- Create your feature branch (git checkout -b feature/your-feature)

- Commit your changes (git commit -am 'Add some feature')

- Push to the branch (git push origin feature/your-feature)

- Open a Pull Request

## Acknowledgments
