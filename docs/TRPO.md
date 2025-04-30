# Trust Region Policy Optimization (TRPO)  
**Type**: *Policy Gradient, On-Policy*  

## What It Is  
TRPO ensures stable policy updates by restricting the size of changes using a mathematical constraint (KL-divergence).

## How It Works  
- **Trust Region**: Limits how far the new policy can deviate from the old one.  
- **Conjugate Gradient Method**: Solves optimization problems under constraints.  

## When to Use It  
- High-risk environments (e.g., autonomous vehicles).  
- Tasks with sparse rewards (e.g., long-term planning).  

## Key Ideas  
- **Safe Updates**: Guarantees monotonic improvement.  
- **Complex**: Requires advanced optimization techniques.  