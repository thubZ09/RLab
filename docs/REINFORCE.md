## REINFORCE (Monte Carlo Policy Gradient)
### Category: Policy Gradient, On-Policy, Monte Carlo

## Core Concept
   Directly learns a policy `πθ(a∣s)` (represented by `PolicyNetwork`) that maps states to action probabilities. It runs a full episode, collects the trajectory (states, actions, rewards), and then updates the policy parameters θ to make actions that led to higher overall returns more likely. It uses the log-derivative trick to estimate the policy gradient.

## Mathematical Formulation
The objective is to maximize the expected total discounted return J(θ). The gradient is estimated using sampled trajectories:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) G_{i,t}
$$

Where:

$$
G_{i,t} = \sum_{k=t}^{T_i} \gamma^{k-t} r_{i,k}
$$

is the discounted return from timestep \( t \) onwards in episode \( i \).

The loss function used in gradient descent is typically the negative of this objective:

$$
\text{Loss} = -\sum_{t} \log \pi_\theta(a_t \mid s_t) G_t
$$

(for one episode).

## When & Where to Use
- Simpler problems or as a baseline. Can handle continuous action spaces (by outputting parameters of a distribution, e.g., mean and std dev for Gaussian).  
- When the policy needs to be stochastic.  
- Suffers from high variance in the gradient estimate because the return Gt depends on the entire future trajectory. This leads to slow convergence.  
- Requires collecting full episodes before updating (Monte Carlo). Updates are infrequent.  
- On-policy: Requires new samples from the current policy for each update.

## Implementation Notes  
 The `ReinforceAgent` uses a `PolicyNetwork`. It stores rewards and log-probabilities of actions taken during an episode. The `learn` method calculates the discounted returns Gt
for each step after the episode finishes and then computes the policy gradient loss to update the network.