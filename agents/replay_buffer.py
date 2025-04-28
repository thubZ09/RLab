import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        # Store data as PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32) # Use float for multiplication later
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Stack tensors to create batch
        return torch.stack(states), torch.stack(actions), torch.stack(rewards), \
               torch.stack(next_states), torch.stack(dones)

    def __len__(self):
        return len(self.buffer)