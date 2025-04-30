# PPO  
**Category** - *Policy Gradient, On-Policy, Trust Region*  

## Core Concept  
- Stabilizes updates via a **clipped surrogate objective** to limit policy changes.  
- **Key techniques**:  
  - **Clipped Probability Ratio**:  
    $$
    r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
    $$  
    Clipped to $[1-\epsilon, 1+\epsilon]$.  

## Mathematical Formulation  
- **Objective**:  
  $$
  \mathcal{L}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
  $$  

## When & Where to Use  
- **Use cases**:  
  - Complex continuous control (e.g., humanoid locomotion).  
  - Transfer from simulation to real-world tasks.  
- **Limitations**:  
  - Requires fresh on-policy data (less sample-efficient).  

## Implementation Notes  
- **Key components**:  
  - `PPOAgent` with actor-critic networks and GAE calculator.  
  - Multiple epochs (4–10) with mini-batch SGD.  
- **Hyperparameters**:  
  - Clipping range ($\epsilon = 0.2$), entropy coefficient ($0.01$).  
- **Optimization**:  
  - Gradient norm clipping (norm < 0.5).  