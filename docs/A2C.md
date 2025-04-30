# A2C  
**Category** - *Policy Gradient, On-Policy, Actor-Critic*  

## Core Concept  
- Combines policy gradient (actor) and value estimation (critic) for lower variance.  
- **Key techniques**:  
  - **Advantage Function**:  
    $$
    A(s,a) = Q(s,a) - V(s)
    $$  
  - **Entropy Regularization**: Encourages exploration by adding entropy to the loss.  

## Mathematical Formulation  
- **Loss Function**:  
  $$
  \mathcal{L} = \underbrace{\mathbb{E}[\log \pi_\theta(a|s) A(s,a)]}_{\text{Actor}} + \underbrace{\beta \mathbb{E}[(V(s) - G_t)^2]}_{\text{Critic}} - \underbrace{\lambda H(\pi)}
  $$  
  - $\beta$: Critic coefficient | $\lambda$: Entropy coefficient  

## When & Where to Use  
- **Use cases**:  
  - Continuous control (e.g., MuJoCo, Pendulum).  
  - Parallelizable environments (synchronous A2C).  
- **Limitations**:  
  - Sensitive to hyperparameters (learning rate, entropy weight).  

## Implementation Notes  
- **Key components**:  
  - `A2CAgent` with shared actor-critic network (or separate).  
  - Use GAE for advantage estimation.  
- **Hyperparameters**:  
  - Actor LR ($5e-4$), critic LR ($1e-3$), $\gamma = 0.99$.  
- **Optimization**:  
  - Normalize advantages: $A \leftarrow (A - \mu)/\sigma$.  