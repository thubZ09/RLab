# REINFORCE  
**Category** - *Policy Gradient, On-Policy*  

## Core Concept  
- **Policy gradient** method that optimizes the policy $\pi_\theta(a|s)$ directly via Monte Carlo returns.  
- **Key techniques**:  
  - **Baseline Subtraction**: Reduces variance by subtracting a state-dependent baseline $b(s)$.  
  - **Softmax Policy**: Used for discrete actions:  
    $$
    \pi_\theta(a|s) = \frac{e^{f_\theta(s,a)}}{\sum_{a'} e^{f_\theta(s,a')}}
    $$  

## Mathematical Formulation  
- **Policy Gradient**:  
  $$
  \nabla J(\theta) = \mathbb{E}\left[ \sum_{t=0}^T G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
  $$  
  - $G_t$: Cumulative return from timestep $t$.  

## When & Where to Use  
- **Use cases**:  
  - Episodic tasks (e.g., CartPole, Acrobot).  
  - Discrete/continuous actions (via Gaussian policies).  
- **Limitations**:  
  - High variance → requires many samples per update.  

## Implementation Notes  
- **Key components**:  
  - `REINFORCEAgent` with policy network and baseline/value network (optional).  
  - Use `Gymnasium`’s `Episode` wrapper for return computation.  
- **Hyperparameters**:  
  - Learning rate ($1e-3$), entropy coefficient ($0.01$).  
- **Optimization**:  
  - Normalize returns: $G_t \leftarrow (G_t - \mu)/\sigma$.  