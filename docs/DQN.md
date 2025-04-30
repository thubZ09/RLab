# DQN  
**Category** - *Value-Based, Off-Policy*  

## Core Concept  
- Extends Q-Learning to large/continuous state spaces using a **neural network** to approximate $Q(s,a)$.  
- **Key techniques**:  
  - **Experience Replay**: Stores transitions $(s,a,r,s',\text{done})$ in a buffer and samples mini-batches.  
  - **Target Network**: Frozen network $Q_{\text{target}}$ computes stable targets:  
    $$
    y_t = r + \gamma (1 - \text{done}) \max_{a'} Q_{\text{target}}(s',a')
    $$  

## Mathematical Formulation  
- **Loss Function**:  
  $$
  \mathcal{L}(\theta) = \mathbb{E}\left[ \left( y_t - Q_{\text{policy}}(s,a;\theta) \right)^2 \right]
  $$  
  - $y_t$: TD target computed using $Q_{\text{target}}$.  

## When & Where to Use  
- **Use cases**:  
  - High-dimensional states (e.g., Atari games from pixels).  
  - Discrete action spaces.  
- **Limitations**:  
  - Cannot handle continuous actions directly.  
  - Overestimates Q-values (addressed by Double DQN).  

## Implementation Notes  
- **Key components**:  
  - `DQNAgent` class with `policy_net`, `target_net`, and `ReplayBuffer`.  
  - Use CNNs for images, MLPs for low-dimensional states.  
- **Hyperparameters**:  
  - Batch size (32–64), replay buffer size (1e5), target update interval (100–1000 steps).  
- **Optimization**:  
  - Use Huber loss for robustness.  
  - Decay ε from 1.0 to 0.1.  