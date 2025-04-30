# REINFORCE  
**Type**: *Policy Gradient, On-Policy*  

## What It Is  
REINFORCE directly learns a policy (a strategy for choosing actions) by adjusting the parameters of a neural network to maximize rewards.

## How It Works  
- **Policy Gradient**: Updates the policy in the direction that increases expected rewards.  
- **Monte Carlo Updates**: Waits until the end of an episode to update the policy based on total rewards.  
- **Baseline Subtraction**: Reduces variance by subtracting a baseline value (e.g., average reward) from returns.  

## When to Use It  
- Episodic tasks (e.g., CartPole, Acrobot).  
- Problems where you need a stochastic policy (e.g., poker).  

## Key Ideas  
- **Policy Optimization**: Learns a direct mapping from states to actions.  
- **High Variance**: Needs many episodes to converge.  