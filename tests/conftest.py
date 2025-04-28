import pytest
import numpy as np
from agents import QLearningAgent, DQNAgent, PPOAgent, TRPOAgent
from environments import GridWorld, ContinuousCartPole

@pytest.fixture
def gridworld():
    return GridWorld(size=5)

@pytest.fixture
def continuous_env():
    return ContinuousCartPole()

@pytest.fixture
def q_agent():
    return QLearningAgent(num_states=25, num_actions=4)

@pytest.fixture
def dqn_agent():
    return DQNAgent(state_dim=4, action_dim=2, batch_size=32)

@pytest.fixture
def ppo_agent():
    return PPOAgent(state_dim=4, action_dim=2)

@pytest.fixture
def trpo_agent():
    return TRPOAgent(state_dim=4, action_dim=2)