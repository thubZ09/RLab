import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

class DPOAgent:
    def __init__(self, context_size=10, action_size=9, 
                 hidden_dim=32, beta=0.1, lr=1e-3):
        """
        Simplified DPO agent for preference learning
        - Context: Integer 0-9
        - Actions: 9 possible sequences (AA, AB, AC, BA, BB, BC, CA, CB, CC)
        """
        self.beta = beta
        
        # policy network
        self.policy = nn.Sequential(
            nn.Embedding(context_size, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size))
        
        # reference network (fixed)
        self.reference = nn.Sequential(
            nn.Embedding(context_size, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size))
        self.reference.load_state_dict(self.policy.state_dict())
        self.reference.requires_grad_(False)
        
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.action_map = self.create_action_map()

    def create_action_map(self):
        """Map action indices to sequence strings"""
        chars = ['A', 'B', 'C']
        return {i: c1+c2 for i, (c1, c2) in enumerate([(a,b) for a in chars for b in chars])}

    def get_log_probs(self, network, contexts, actions):
        """Get log probabilities for (context, action) pairs"""
        context_emb = network[0](contexts)
        logits = network[1](context_emb).relu_()
        logits = network[3](logits)
        return torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1))

    def compute_loss(self, contexts, y_w, y_l):
        """Compute DPO loss for batch of preferences"""
        # get log probabilities
        policy_logp_w = self.get_log_probs(self.policy, contexts, y_w)
        policy_logp_l = self.get_log_probs(self.policy, contexts, y_l)
        ref_logp_w = self.get_log_probs(self.reference, contexts, y_w)
        ref_logp_l = self.get_log_probs(self.reference, contexts, y_l)
        
        # compute log ratios
        log_ratio_w = policy_logp_w - ref_logp_w
        log_ratio_l = policy_logp_l - ref_logp_l
        
        # loss
        losses = -torch.nn.functional.logsigmoid(self.beta * (log_ratio_w - log_ratio_l))
        return losses.mean()

    def update(self, contexts, y_w, y_l):
        """Perform update step"""
        self.optimizer.zero_grad()
        loss = self.compute_loss(contexts, y_w, y_l)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def generate_preference_data(num_samples=1000):
    """Generate synthetic preference data based on rules"""
    data = []
    action_dict = {'AA':0, 'AB':1, 'AC':2, 'BA':3, 'BB':4, 'BC':5, 'CA':6, 'CB':7, 'CC':8}
    
    for _ in range(num_samples):
        context = np.random.randint(10)
        
        # create preference pairs based on context
        if context < 5:
            y_w = 'AA'  # Preferred
            y_l = np.random.choice(['AB', 'AC', 'BA'])  # Dispreferred
        else:
            y_w = 'BC'  # Preferred
            y_l = np.random.choice(['BA', 'BB', 'CA'])  # Dispreferred
        
        data.append((
            torch.tensor(context),
            torch.tensor(action_dict[y_w]),
            torch.tensor(action_dict[y_l])
        ))
    return data

def evaluate_policy(agent, num_tests=100):
    """Evaluate policy against preference rules"""
    results = defaultdict(int)
    action_map = agent.action_map
    
    for _ in range(num_tests):
        context = torch.tensor(np.random.randint(10))
        
        # Get policy action
        with torch.no_grad():
            logits = agent.policy(context)
            policy_action = torch.argmax(logits).item()
        
        # Get reference action
        with torch.no_grad():
            logits = agent.reference(context)
            ref_action = torch.argmax(logits).item()
        
        # Check rule compliance
        context = context.item()
        policy_seq = action_map[policy_action]
        ref_seq = action_map[ref_action]
        
        if context < 5:
            results['policy_correct'] += int(policy_seq == 'AA')
            results['ref_correct'] += int(ref_seq == 'AA')
        else:
            results['policy_correct'] += int(policy_seq == 'BC')
            results['ref_correct'] += int(ref_seq == 'BC')
    
    accuracy = lambda x: x / num_tests * 100
    print(f"Policy Accuracy: {accuracy(results['policy_correct']):.1f}%")
    print(f"Reference Accuracy: {accuracy(results['ref_correct']):.1f}%")

def train_dpo():
    # initialize agent and data
    agent = DPOAgent()
    data = generate_preference_data(num_samples=1000)
    
    # convert to tensors
    contexts = torch.stack([d[0] for d in data])
    y_w = torch.stack([d[1] for d in data])
    y_l = torch.stack([d[2] for d in data])
    
    # training loop
    batch_size = 32
    num_epochs = 20
    
    print("Initial Evaluation:")
    evaluate_policy(agent)
    
    for epoch in range(num_epochs):
        permutation = torch.randperm(len(data))
        total_loss = 0
        
        for i in range(0, len(data), batch_size):
            batch_idx = permutation[i:i+batch_size]
            batch_loss = agent.update(
                contexts[batch_idx],
                y_w[batch_idx],
                y_l[batch_idx]
            )
            total_loss += batch_loss
            
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Avg Loss: {total_loss / (len(data)/batch_size):.4f}")
            evaluate_policy(agent)

if __name__ == "__main__":
    train_dpo()