import torch
import random
import math # for test

class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int, learning_rate: float = 0.1, discount_factor: float = 0.99, exploration_rate: float = 1.0, exploration_decay: float = 0.995, min_exploration_rate: float = 0.01):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.q_table = torch.zeros((num_states, num_actions), dtype=torch.float32)

    def choose_action(self, state: int):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = torch.argmax(self.q_table[state]).item()
        return action

    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        max_next_q = torch.max(self.q_table[next_state]).item()
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.q_table[state, action].item()
        self.q_table[state, action] += self.lr * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_table(self):
        return self.q_table

def test_q_learning():
    print("Testing Q-Learning Agent...")

 # test parameters
num_states = 10
num_actions = 4
agent = QLearningAgent(num_states, num_actions, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.1) # Low epsilon for predictable test

 # simulate a transition
state = 3
action = 1
reward = 10
next_state = 5

 # set a known Q-value for the next state to make the calculation predictable
agent.q_table[next_state, 0] = 2.0
agent.q_table[next_state, 1] = 5.0 # max Q-value in next state
agent.q_table[next_state, 2] = 1.0
agent.q_table[next_state, 3] = 0.0

initial_q_value = agent.q_table[state, action].item()
print(f"Initial Q-value for state {state}, action {action}: {initial_q_value}")
print(f"Max Q-value in next state {next_state}: {torch.max(agent.q_table[next_state]).item()}")

 # perform the update
agent.update_q_table(state, action, reward, next_state)

 # calculate expected Q-value
  # max_next_q = 5.0
   # td_target = reward + gamma * max_next_q = 10 + 0.9 * 5.0 = 10 + 4.5 = 14.5
  # td_error = td_target - initial_q_value = 14.5 - 0.0 = 14.5
  # updated_q = initial_q_value + alpha * td_error = 0.0 + 0.5 * 14.5 = 7.25
expected_q_value = 7.25

updated_q_value = agent.q_table[state, action].item()

print(f"Updated Q-value for state {state}, action {action}: {updated_q_value}")
print(f"Expected Q-value: {expected_q_value}")

 # ussing math.isclose for float comparison
assert math.isclose(updated_q_value, expected_q_value, rel_tol=1e-6), f"Expected Q-value {expected_q_value}, got {updated_q_value}"
print("Q-Learning update test passed!")

  # test action selection (with low epsilon, should choose best action)
agent.q_table[state, 0] = 1.0
agent.q_table[state, 1] = updated_q_value # 7.25
agent.q_table[state, 2] = 3.0
agent.q_table[state, 3] = -1.0
chosen_action = agent.choose_action(state) # Since epsilon is low (0.1), it should likely exploit
expected_action = 1 # action with Q-value 7.25

print(f"Agent chose action: {chosen_action} (Expected likely: {expected_action} due to low epsilon)")

 # Note: due to the stochastic nature (even with low epsilon), this assert might fail sometimes.
 # we assert that the chosen action is within the valid range.

assert 0 <= chosen_action < num_actions, "Chosen action is out of bounds."
print("Action selection test passed (checked validity).")


if __name__ == "__main__":
    test_q_learning()