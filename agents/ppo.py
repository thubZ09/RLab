import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, clip_eps=0.2, epochs=10,
                 batch_size=64, lambda_gae=0.95):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_gae = lambda_gae

        # actor network (policy)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def choose_action(self, state):
        """Returns action and log probability of that action under current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            probs = self.policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_advantages_and_returns(self, rewards, dones, states, next_states):
        """Computes GAE advantages and returns using critic network"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            next_states_tensor = torch.FloatTensor(next_states)
            values = self.critic(states_tensor).squeeze().numpy()
            next_values = self.critic(next_states_tensor).squeeze().numpy()
        
        advantages = []
        returns = []
        gae = 0
        next_value = 0  
        
        # eeverse iteration through transitions
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                next_value = 0
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def update(self, states, actions, old_log_probs, advantages, returns):
        """Performs PPO update with clipped objective"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        for _ in range(self.epochs):
            # actor update
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            # Critic update
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

def train_ppo():
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Hyperparameters
    num_episodes = 500
    max_steps = 500
    batch_size = 2000
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        clip_eps=0.2,
        epochs=10,
        batch_size=batch_size
    )
    
    # training loop
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        transitions = []
        
        # collect trajectories
        for _ in range(max_steps):
            action, log_prob = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # process collected data
        states = [t['state'] for t in transitions]
        actions = [t['action'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        next_states = [t['next_state'] for t in transitions]
        dones = [t['done'] for t in transitions]
        old_log_probs = [t['log_prob'] for t in transitions]
        
        # compute advantages and returns
        advantages, returns = agent.compute_advantages_and_returns(
            rewards, dones, states, next_states
        )
        
        # update networks
        agent.update(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages.numpy(),
            returns=returns.numpy()
        )
        
        # track performance
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Average Reward (last 10): {avg_reward:.1f}")
    
    print("Training completed!")
    env.close()

if __name__ == "__main__":
    train_ppo()