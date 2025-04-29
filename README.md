# RLab (PyTorch)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLab is a repository dedicated to exploring, understanding, and implementing various reinforcement learning algorithms in PyTorch. It provides clear, concise implementations suitable for educational purposes, experimentation, and as a foundation for further research in RL.

## 📌Goals

* Provide understandable implementations of key RL algorithms, from classic to modern approaches.
* Facilitate comparison and experimentation between different RL paradigms.
* Serve as a practical learning resource for practitioners entering the field of RL.
* Offer a modular structure that can be extended for custom agents, environments, and projects.


### ✅**Implemented Algorithms**
* **Value-Based:**
    * Q-Learning
    * SARSA
    * Deep Q-Network (DQN)
* **Policy Gradient:**
    * REINFORCE (Monte Carlo Policy Gradient)
    * Advantage Actor-Critic (A2C)
    * Proximal Policy Optimization (PPO)
    * Trust Region Policy Optimization (TRPO)
* **Preference-Based:**
    * Direct Preference Optimization (DPO)

### ✅**Key Components**
* **Algorithm Implementations:** Clear implementations of foundational and widely-used RL algorithms.
* **Custom Environments:** Example environments (Gridworld, simple continuous tasks) for testing and development.
* **Example Scripts:** Ready-to-run examples demonstrating how to train agents in standard environments (e.g., CartPole) or with specific techniques (e.g., preference learning).
* **Testing Suite:** Unit tests to ensure the correctness of agent logic and environment dynamics.
* **Detailed Documentation:** In-depth explanations for each implemented algorithm, covering concepts, mathematics, and usage guidelines.


## 📌Getting Started
### Prerequisites
* Python 3.8+
* Pip (Python package installer)

### Installation

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/RLab.git](https://github.com/your-username/RLab.git) # <-- Replace with your repo URL!
    cd RLab
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📌Usage
* **Run Examples:** The easiest way to see the agents in action is to run the example scripts located in the `examples/` directory.
    ```bash
    # Example: Train PPO on CartPole
    python examples/cartpole_ppo.py

    # Example: Demonstrate DPO (may require specific setup/data)
    python examples/preference_dpo.py
    ```
* **Use in your projects:** The `agents` and `environments` modules are designed to be importable. You can integrate them into your own RL projects or research.

## 📌Documentation

Detailed explanations for each algorithm, including their core concepts, mathematical formulations, use cases, and implementation notes, can be found in the `docs/` directory. Refer to the corresponding markdown file for each agent.

## 📌Algorithm Selection Guide

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

## 📌Contributing  
Contributions to RLab are welcome! If you find bugs, have suggestions for improvements, or want to add new algorithms or features, please feel free to:

- Open an Issue to discuss the change.  
- Fork the repository and submit a PR with your contribution.  

Please ensure your code follows the existing style and includes appropriate tests and documentation.

## 📌Acknowledgments
- [OpenAI Gym/Gymnasium documentation](https://github.com/openai/gym)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/en/latest/)
- [RL Baselines3 Zoo - For reference implementations (Note: uses different structure)](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
- [Reinforcement Learning Overview by Kevin Murphy](https://arxiv.org/pdf/2412.05265)
- Original research papers for specific algorithms (DQN, PPO, DPO....)
