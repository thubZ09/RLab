import torch
import random
import math

class SarsaAgent:
    def __init__(self, num_states: int, num_actions: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.99, exploration_rate: float = 1.0, 
                 exploration_decay: float = 0.995, min_exploration_rate: float = 0.01):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.q_table = torch.zeros((num_states, num_actions), dtype=torch.float32)

    def choose_action(self, state: int):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return torch.argmax(self.q_table[state]).item()

    def update_q_table(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int):
        next_q = self.q_table[next_state, next_action].item()
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state, action].item()
        self.q_table[state, action] += self.lr * td_error
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def test_sarsa():
    print("Testing SARSA Agent...")
    agent = SarsaAgent(10, 4, learning_rate=0.5, exploration_rate=0.1)
    
    state = 3
    action = 1
    reward = 10
    next_state = 5
    next_action = 0
    
    agent.q_table[next_state] = torch.tensor([2.0, 5.0, 1.0, 0.0])
    agent.update_q_table(state, action, reward, next_state, next_action)
    
    expected_q = 0 + 0.5 * (10 + 0.99*2.0 - 0)
    assert math.isclose(agent.q_table[3,1].item(), expected_q, rel_tol=1e-6)
    print("SARSA test passed!")

if __name__ == "__main__":
    test_sarsa()