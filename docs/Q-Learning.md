# Q-Learning  
**Type**: *Value-Based, Off-Policy* 

## What It Is  
Q-Learning is a basic reinforcement learning algorithm that teaches an agent to make decisions by learning the value of actions in specific states. It builds a table (Q-table) that maps state-action pairs to expected future rewards.

## How It Works  
- **Q-Table**: A lookup table where each entry represents "how good" an action is in a given state.  
- **Update Rule**:  
  When the agent observes a reward and a new state, it updates the Q-value for the previous state-action pair using:  
  `New Q-value = Old Q-value + Learning Rate * (Reward + Discount Factor * Best Future Reward - Old Q-value)`  
- **Exploration vs. Exploitation**: Uses ε-greedy strategy (e.g., pick random actions sometimes to explore).  

## When to Use It  
- Small, discrete environments (e.g., GridWorld, FrozenLake).  
- Problems where you can represent states and actions as numbers.  

## Key Ideas  
- **Model-Free**: Doesn’t need to know how the environment works.  
- **Off-Policy**: Learns optimal behavior even if the agent isn’t following the optimal policy.  