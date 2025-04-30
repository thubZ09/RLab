# TRPO  
**Category** - *Policy Gradient, On-Policy, Trust Region*  

## Core Concept  
- Enforces policy updates via **KL-divergence constraints** for guaranteed improvement.  
- **Key techniques**:  
  - **Constrained Optimization**:  
    $$
    \max_\theta L_{\text{surrogate}}(\theta) \quad \text{s.t. } D_{\text{KL}}(\theta, \theta_{\text{old}}) \leq \delta
    $$  

## Mathematical Formulation  
- **Conjugate Gradient Method**: Solves for the search direction under the KL constraint.  
- **Fisher Information Matrix**: Approximates curvature:  
  $$
  F = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta \nabla_\theta \log \pi_\theta^T \right]
  $$  

## When & Where to Use  
- **Use cases**:  
  - High-risk environments (e.g., autonomous driving).  
  - Sparse/rewarding tasks.  
- **Limitations**:  
  - Computationally intensive (Hessian-vector products).  

## Implementation Notes  
- **Key components**:  
  - `TRPOAgent` with conjugate gradient solver.  
  - Line search for step-size adaptation.  
- **Hyperparameters**:  
  - Trust region radius ($\delta = 0.01$).  
- **Optimization**:  
  - Use Hessian-free optimization.  