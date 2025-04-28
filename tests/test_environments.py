import pytest
import numpy as np

def test_gridworld_reset(gridworld):
    state = gridworld.reset()
    assert state == (0, 0)
    assert gridworld.state == (0, 0)

def test_gridworld_step(gridworld):
    gridworld.reset()
    # Test valid movement
    new_state, reward, done, _ = gridworld.step(1)  # Move down
    assert new_state == (1, 0)
    assert not done
    
    # Test obstacle collision
    gridworld.state = (2, 1)
    _, reward, done, _ = gridworld.step(0)  # Move up into obstacle
    assert reward == -1.0
    assert done

def test_gridworld_goal(gridworld):
    gridworld.state = (4, 3)
    _, reward, done, _ = gridworld.step(3)  # Move right to goal
    assert reward == 1.0
    assert done

def test_continuous_env_action_conversion(continuous_env):
    continuous_action = -0.5
    obs, reward, done, _ = continuous_env.step([continuous_action])
    assert not done  # Shouldn't terminate immediately
    assert isinstance(reward, float)

def test_continuous_env_observations(continuous_env):
    obs = continuous_env.reset()
    assert obs.shape == (4,)
    obs, _, _, _ = continuous_env.step([0.5])
    assert obs.shape == (4,)