import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from agents.trpo import TRPOAgent

# setup
def create_policy_net(state_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, action_dim),
        nn.Softmax(dim=-1)
    )

def create_value_net(state_dim):
    return nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

def compute_gae(rewards, dones, values, next_values, gamma=0.99, lambda_gae=0.95):
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    next_value = next_values[-1]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - dones[t]) * next_values[t] - values[t]
        advantages[t] = last_advantage = delta + gamma * lambda_gae * (1 - dones[t]) * last_advantage
        next_value = values[t]
    returns = advantages + values
    return advantages, returns

# training Setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = create_policy_net(state_dim, action_dim)
value_net = create_value_net(state_dim)
agent = TRPOAgent(policy_net, value_net, gamma=0.99)

# hyperparameters
num_episodes = 500
batch_size = 2000
gamma = 0.99
lambda_gae = 0.95

# training Loop
memory = []
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    transitions = []
    
    # trajectories
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))
            value = value_net(state_tensor).item()
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        transitions.append((state, action, reward, next_state, done, action_probs[action].item(), value))
        
        state = next_state
        episode_reward += reward
    
    # process collected transitions
    states, actions, rewards, next_states, dones, old_probs, values = zip(*transitions)
    next_values = [value_net(torch.FloatTensor(s)).item() for s in next_states] + [0]
    
    # compute GAE and returns
    advantages, returns = compute_gae(
        np.array(rewards),
        np.array(dones),
        np.array(values),
        np.array(next_values[1:]),
        gamma,
        lambda_gae
    )
    
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # update networks
    agent.update(
        states=np.array(states),
        actions=np.array(actions),
        advantages=advantages,
        old_probs=np.array(old_probs),
        returns=np.array(returns)
    
    episode_rewards.append(episode_reward))
    
    # logging
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode+1}, Last 10 Avg Reward: {avg_reward:.1f}")

print("Training completed!")

# save Policy
torch.save(policy_net.state_dict(), "trpo_cartpole.pth")