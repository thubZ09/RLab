# DPO  
**Category** - *Preference-Based, Off-Policy*  

## Core Concept  
- Converts human preferences into policies without explicit reward functions.  
- **Key techniques**:  
  - **Closed-Form Update**:  
    $$
    \pi_{\text{opt}}(a|s) \propto \pi_{\text{ref}}(a|s) \exp\left( \frac{\beta}{2} \log \frac{D(s,a)}{1-D(s,a)} \right)
    $$  
    where $D(s,a)$ is the preference dataset.  

## Mathematical Formulation  
- **Loss Function**:  
  $$
  \mathcal{L}(\theta) = -\mathbb{E}_{(s,a^+,a^-) \sim D} \left[ \log \sigma\left( \beta (\log \pi_\theta(a^+|s) - \log \pi_\theta(a^-|s)) \right) \right]
  $$  
  - $\sigma$: Sigmoid function | $\beta$: Temperature parameter  

## When & Where to Use  
- **Use cases**:  
  - Alignment with human values (e.g., LLMs, robotics).  
  - No explicit reward signal available.  
- **Limitations**:  
  - Requires pre-collected preference dataset.  

## Implementation Notes  
- **Key components**:  
  - `DPOAgent` with policy network and preference buffer.  
  - Use rejection sampling for iterative refinement.  
- **Hyperparameters**:  
  - Temperature ($\beta = 0.1$), batch size (256).  
- **Optimization**:  
  - Use Adam optimizer with LR ($5e-4$).  