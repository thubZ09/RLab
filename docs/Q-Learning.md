# Q-Learning  
**Category** - *Value-Based, Off-Policy*  

## Core Concept  
- **Model-free** algorithm that learns optimal Q-values $Q(s,a)$ for discrete state-action spaces.  
- **Key techniques**:  
  - **Bellman Equation**: Updates Q-values iteratively using:  
    $$
    Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
    $$  
  - **ε-Greedy Exploration**: Balances exploration/exploitation by choosing greedy actions with probability $1-\epsilon$ and random actions with probability $\epsilon$.  

## Mathematical Formulation  
- **Q-Value Update**:  
  $$
  Q(s,a) \leftarrow Q(s,a) + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD Target}} - Q(s,a) \right]
  $$  
  - $\alpha$: Learning rate | $\gamma$: Discount factor | $s'$: Next state  

## When & Where to Use  
- **Use cases**:  
  - Small, discrete state-action spaces (e.g., GridWorld, FrozenLake).  
  - Environments with deterministic transitions.  
- **Limitations**:  
  - Scales poorly to large/continuous spaces due to table storage.  

## Implementation Notes  
- **Key components**:  
  - `QAgent` class manages the Q-table and ε-greedy strategy.  
  - Use a `numpy.ndarray` for Q-table storage.  
- **Hyperparameters**:  
  - Learning rate ($\alpha = 0.1$), discount factor ($\gamma = 0.99$), ε bounds ($0.01 \leq \epsilon \leq 0.5$).  
- **Optimization**:  
  - Decay ε over time (e.g., exponential decay).  