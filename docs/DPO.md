# Direct Preference Optimization (DPO)  
**Type**: *Preference-Based, Off-Policy*  

## What It Is  
DPO uses human preferences (e.g., "A is better than B") to train policies without explicit reward functions.

## How It Works  
- **Preference Dataset**: Collects pairs of trajectories where one is preferred over another.  
- **Closed-Form Update**: Directly optimizes the policy to align with preferences.  

## When to Use It  
- Alignment with human values (e.g., chatbots, robotics).  
- No explicit reward signal available.  

## Key Ideas  
- **Preference-Based**: Learns from comparisons, not rewards.  
- **Efficient**: Avoids reward modeling and reinforcement learning steps.  