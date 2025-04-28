## Deep Q-Network (DQN)

### Category - **Value-Based, Off-Policy**

## Core Concept
 Extends Q-Learning to handle large/continuous state spaces (like pixels from a screen) by approximating the Q-function Q(s,a) using a neural network (the QNetwork). Uses two key techniques:
- **Experience Replay**: Stores past transitions (state, action, reward, next_state, done) in a `ReplayBuffer`. During learning, it samples random mini-batches from this buffer. This breaks correlations between consecutive samples and reuses data efficiently.
- **Target Network**: A separate `target_net` is used, whose weights are frozen for several steps and periodically updated with the `policy_net` weights. 

The `target_net` calculates the target Q-values:

$$
r + \gamma \max_{a'} Q_{\text{target}}(s', a')
$$

These target Q-values are used in the loss calculation. This approach adds stability compared to using the constantly changing `policy_net` for targets.

## Mathematical Formulation
The loss function aims to minimize the difference between the predicted Q-value and the target Q-value (Temporal Difference error), typically using Mean Squared Error (MSE) or Huber loss. 

$$
\text{Loss} = \Big( 
\underbrace{\text{TD Target}}_{\text{r} + \gamma \max_{a'} Q_{\text{target}}(s', a')}
- 
\underbrace{\text{Predicted Q}}_{Q_{\text{policy}}(s, a)}
\Big)^2
$$

## When & Where to Use
- Problems with large, discrete action spaces and large/continuous state spaces (e.g., Atari games from pixels).
- Can suffer from overestimation of Q-values (addressed by Double DQN).
- When sample efficiency is important (due to experience replay).
- Doesn't directly handle continuous action spaces.

## Implementation Notes
The `DQNAgent` class manages the policy and target networks, the replay buffer, and the epsilon-greedy exploration strategy.   The `learn` method samples a batch, calculates predicted and target Q-values, computes the loss, and performs gradient descent.