import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma

    def choose_action(self, state):
        probs = self.actor(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, transitions):
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.stack([torch.FloatTensor(s) for s in states])
        next_states = torch.stack([torch.FloatTensor(s) for s in next_states])
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        
        # ccritic update
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze().detach()
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        critic_loss = F.mse_loss(values, td_targets)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # actor update
        advantages = td_targets.detach() - values.detach()
        log_probs = torch.log(self.actor(states).gather(1, torch.LongTensor(actions).unsqueeze(1)))
        actor_loss = -(log_probs * advantages.unsqueeze(1)).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

def test_actor_critic():
    agent = ActorCriticAgent(4, 2)
    transitions = [
        (np.array([0.1,0.2,0.3,0.4]), 0, 1.0, np.array([0.2,0.3,0.4,0.5]), False)
    ]
    agent.update(transitions)
    print("Actor-Critic test passed!")

if __name__ == "__main__":
    test_actor_critic()