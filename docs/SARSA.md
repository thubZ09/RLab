# SARSA  
**Type**: *Value-Based, On-Policy*  

## What It Is  
SARSA is a Q-learning variant that learns action values based on the agent’s actual behavior (not hypothetical best actions). It stands for State-Action-Reward-State-Action.

## How It Works  
- **On-Policy Updates**:  
  Updates Q-values using actions the agent actually took, not the best possible ones.  
  `New Q-value = Old Q-value + Learning Rate * (Reward + Discount Factor * Next Action's Q-value - Old Q-value)`  
- **Safer Exploration**: Since it uses real actions, it avoids risky moves that Q-learning might suggest.  

## When to Use It  
- Environments where exploration matters (e.g., robotics).  
- Stochastic (random) environments where outcomes are uncertain.  

## Key Ideas  
- **On-Policy**: Learns from the same policy it uses to act.  
- **Safer**: Avoids overestimating rewards because it uses real next actions.  