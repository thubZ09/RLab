import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ContinuousCartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        # Override action space to be continuous
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = self.env.observation_space
        
    def reset(self):
        return self.env.reset()[0]
    
    def step(self, action):
        # convert continuous action to discrete
        discrete_action = 0 if action < 0 else 1
        obs, reward, done, trunc, info = self.env.step(discrete_action)
        return obs, reward, done, info
    
    def render(self):
        self.env.render()