import numpy as np

class GridWorld:
    def __init__(self, size=5, start=(0,0), goal=(4,4), obstacles=[(2,2)]):
        """
        Creates grid world with:
        - size: Grid dimension (size x size)
        - start: Starting position
        - goal: Terminal state with +1 reward
        - obstacles: Blocked cells with -1 reward
        """
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x,y = self.state
        reward = -0.01  # Step penalty
        
        # Movement logic
        if action == 0: x = max(x-1, 0)          # Up
        elif action == 1: x = min(x+1, self.size-1)  # Down
        elif action == 2: y = max(y-1, 0)        # Left
        elif action == 3: y = min(y+1, self.size-1)  # Right
        
        new_state = (x,y)
        
        # Reward logic
        if new_state == self.goal:
            reward = 1.0
            done = True
        elif new_state in self.obstacles:
            reward = -1.0
            done = True
        else:
            done = False
            
        self.state = new_state
        return new_state, reward, done, {}
    
    def render(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[self.start] = 'S'
        grid[self.goal] = 'G'
        for obs in self.obstacles: grid[obs] = 'X'
        print(grid)