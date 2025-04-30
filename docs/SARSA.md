# SARSA  
**Category** - *Value-Based, On-Policy*  

## Core Concept  
- **Model-free** algorithm that learns Q-values using the **actual policy** (on-policy).  
- **Key techniques**:  
  - **TD Update with Current Policy**:  
    $$
    Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]
    $$  
    where $a'$ is selected using the current policy (e.g., ε-greedy).  

## Mathematical Formulation  
- **Q-Value Update**:  
  $$
  Q(s,a) \leftarrow Q(s,a) + \alpha \left[ \underbrace{r + \gamma Q(s',a')}_{\text{TD Target}} - Q(s,a) \right]
  $$  
  - $a'$: Action taken in state $s'$ (unlike Q-Learning’s $\max$).  

## When & Where to Use  
- **Use cases**:  
  - Safety-critical environments (e.g., robotics).  
  - Stochastic environments where exploration matters.  
- **Limitations**:  
  - Requires on-policy data → less sample-efficient than off-policy methods.  

## Implementation Notes  
- **Key components**:  
  - `SARSAAgent` class tracks current policy (e.g., ε-greedy).  
  - Store transitions $(s,a,r,s',a')$ in a buffer.  
- **Hyperparameters**:  
  - Same as Q-Learning (α, γ, ε).  
- **Optimization**:  
  - Use ε-greedy during training (not decayed).  