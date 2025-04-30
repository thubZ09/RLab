# Advantage Actor-Critic (A2C)  
**Type**: *Policy Gradient, On-Policy*  

## What It Is  
A2C combines policy gradient (Actor) and value estimation (Critic) to reduce variance. The Actor chooses actions, and the Critic evaluates how good those actions are.

## How It Works  
- **Actor-Critic Framework**:  
  - **Actor**: Learns the policy (which actions to take).  
  - **Critic**: Learns to estimate state values (how good a state is).  
- **Advantage Function**: Measures how much better an action is compared to the average action in that state.  
- **Entropy Regularization**: Encourages exploration by penalizing overly confident policies.  

## When to Use It  
- Continuous control tasks (e.g., robotic arm movement).  
- Environments where you need stable and fast learning.  

## Key Ideas  
- **Actor-Critic**: Two networks work together for faster learning.  
- **Advantage**: Focuses on actions that outperform the average.  