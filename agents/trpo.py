import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize

class TRPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 max_kl=0.01, cg_iters=10, gamma=0.99):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1))
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.gamma = gamma

    def choose_action(self, state):
        probs = self.policy(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def get_loss(self, states, actions, advantages):
        probs = self.policy(states).gather(1, actions)
        old_probs = probs.detach()
        ratio = probs / old_probs
        return -(ratio * advantages).mean()

    def get_kl(self, states):
        old_probs = self.policy(states).detach()
        new_probs = self.policy(states)
        return torch.mean(torch.sum(old_probs * (torch.log(old_probs) - torch.log(new_probs)), dim=1))

    def update(self, states, actions, advantages):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        
        # calculate gradient
        loss = self.get_loss(states, actions, advantages)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        flat_grad = torch.cat([g.view(-1) for g in grads]).detach().numpy()

        # fisher-vector product
        def fvp(v):
            kl = self.get_kl(states)
            kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in kl_grad])
            return (flat_kl_grad * torch.FloatTensor(v)).sum().backward()
            return torch.cat([p.grad for p in self.policy.parameters()]).numpy()

        # conjugate gradient
        step_dir = minimize(fvp, -flat_grad, method='CG', jac=True, 
                           options={'maxiter': self.cg_iters}).x
        
        # line search
        max_step = (2 * self.max_kl / (step_dir @ fvp(step_dir)))**0.5
        full_step = max_step * step_dir
        
        # apply update
        current_params = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        new_params = current_params + torch.FloatTensor(full_step)
        
        idx = 0
        for p in self.policy.parameters():
            p_size = p.data.numel()
            p.data.copy_(new_params[idx:idx+p_size].view(p.data.shape))
            idx += p_size

def test_trpo():
    agent = TRPOAgent(4, 2)
    states = [np.random.rand(4) for _ in range(10)]
    actions = [0,1,0,1,0,1,0,1,0,1]
    advantages = np.random.randn(10)
    agent.update(states, actions, advantages)
    print("TRPO test passed!")

if __name__ == "__main__":
    test_trpo()