import torch
import numpy as np

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99):
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=-1))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.episode = []
    
    def choose_action(self, state):
        probs = self.policy(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def store_transition(self, state, action, reward):
        self.episode.append((state, action, reward))
    
    def update(self):
        returns = []
        G = 0
        for _, _, r in reversed(self.episode):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for (state, action, _), G in zip(self.episode, returns):
            prob = self.policy(torch.FloatTensor(state))[action]
            policy_loss.append(-torch.log(prob) * G)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.episode = []

def test_reinforce():
    print("Testing REINFORCE...")
    agent = REINFORCEAgent(4, 2)
    agent.store_transition([0.1, 0.2, 0.3, 0.4], 1, 1.0)
    agent.update()
    print("REINFORCE update completed")

if __name__ == "__main__":
    test_reinforce()