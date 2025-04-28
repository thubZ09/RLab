import torch
import random
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim))
        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.target_update = target_update
        self.steps = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.policy_net[-1].out_features-1)
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state)).argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.memory.push(torch.FloatTensor(state), torch.tensor([action]), 
                        torch.FloatTensor(next_state), torch.tensor([reward]), 
                        torch.tensor([done], dtype=torch.bool))
        
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.cat(batch.done)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_values
        
        loss = torch.nn.functional.mse_loss(q_values, expected_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps / self.epsilon_decay)
        self.steps += 1
        
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def test_dqn():
    print("Testing DQN Agent...")
    agent = DQNAgent(4, 2, batch_size=2)
    agent.update([0.1, 0.2, 0.3, 0.4], 0, 1.0, [0.2, 0.3, 0.4, 0.5], False)
    print("DQN update completed")

if __name__ == "__main__":
    test_dqn()