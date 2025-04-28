import pytest
import torch
import numpy as np
from agents import (
    QLearningAgent, SarsaAgent, DQNAgent, 
    REINFORCEAgent, ActorCriticAgent, PPOAgent, TRPOAgent
)

def test_q_learning_initialization(q_agent):
    assert q_agent.q_table.shape == (25, 4)
    assert q_agent.epsilon == 1.0

def test_q_learning_action_selection(q_agent):
    state = 0
    action = q_agent.choose_action(state)
    assert 0 <= action < 4

def test_q_learning_update(q_agent):
    initial_q = q_agent.q_table[0, 0].item()
    q_agent.update_q_table(0, 0, 1.0, 1)
    assert q_agent.q_table[0, 0].item() != initial_q

def test_dqn_buffer_management(dqn_agent):
    # Test buffer capacity
    for _ in range(150):
        dqn_agent.memory.push(
            torch.rand(4), torch.tensor([0]), 
            torch.rand(4), torch.tensor([1.0]), 
            torch.tensor([False]))
    assert len(dqn_agent.memory) == 100

def test_dqn_learning_step(dqn_agent):
    # Populate buffer
    for _ in range(50):
        dqn_agent.update(np.random.rand(4), 0, 1.0, np.random.rand(4), False)
    # Initial params
    initial_params = [p.clone() for p in dqn_agent.policy_net.parameters()]
    dqn_agent.learn()
    # Verify params changed
    for init, new in zip(initial_params, dqn_agent.policy_net.parameters()):
        assert not torch.equal(init, new)

def test_reinforce_update():
    agent = REINFORCEAgent(4, 2)
    agent.store_transition([0.1,0.2,0.3,0.4], 1, 1.0)
    initial_params = [p.clone() for p in agent.policy.parameters()]
    agent.update()
    for init, new in zip(initial_params, agent.policy.parameters()):
        assert not torch.equal(init, new)

def test_actor_critic_update():
    agent = ActorCriticAgent(4, 2)
    transitions = [(np.random.rand(4), 0, 1.0, np.random.rand(4), False)]
    actor_params = [p.clone() for p in agent.actor.parameters()]
    critic_params = [p.clone() for p in agent.critic.parameters()]
    agent.update(transitions)
    # Check both networks updated
    for init, new in zip(actor_params, agent.actor.parameters()):
        assert not torch.equal(init, new)
    for init, new in zip(critic_params, agent.critic.parameters()):
        assert not torch.equal(init, new)

def test_ppo_action_range(ppo_agent):
    action = ppo_agent.choose_action(np.random.rand(4))
    assert action in {0, 1}

def test_trpo_policy_update(trpo_agent):
    states = [np.random.rand(4) for _ in range(10)]
    actions = np.random.randint(0, 2, size=10)
    advantages = np.random.randn(10)
    initial_params = [p.data.clone() for p in trpo_agent.policy.parameters()]
    trpo_agent.update(states, actions, advantages)
    # Verify parameters changed
    for init, new in zip(initial_params, trpo_agent.policy.parameters()):
        assert not torch.equal(init, new.data)