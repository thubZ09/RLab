import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import logsigmoid

class DPOAgent:
    def __init__(self, vocab_size=100, embedding_dim=32, 
                 beta=0.1, lr=1e-4, max_length=20):
        """
        Simplified DPO implementation for preference learning
        Args:
            vocab_size: Size of vocabulary (for synthetic example)
            embedding_dim: Dimension of embeddings
            beta: Temperature parameter for DPO loss
            lr: Learning rate
            max_length: Maximum sequence length for completions
        """
        self.beta = beta
        self.max_length = max_length
        
        # policy network (to be trained)
        self.policy = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, vocab_size))
        
        # reference network (fixed)
        self.reference = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, vocab_size))
        
        # initialize reference with policy weights
        self.reference.load_state_dict(self.policy.state_dict())
        self.reference.requires_grad_(False)  # Freeze reference
        
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)

    def get_log_probs(self, network, sequences):
        """Compute log probabilities for sequences under given network"""
        log_probs = []
        for seq in sequences:
            embeds = network[0](seq)  # Embedding layer
            logits = network[1](embeds.mean(dim=0))  # Average pooling
            log_probs.append(torch.log_softmax(logits, dim=-1))
        return torch.stack(log_probs)

    def compute_loss(self, y_w, y_l, x):
        """
        Compute DPO loss for preferred (y_w) vs dispreferred (y_l) completions
        Args:
            y_w: List of preferred completion sequences (tensors)
            y_l: List of dispreferred completion sequences (tensors)
            x: Input contexts (tensors)
        """
        # get policy log probabilities
        policy_logp_w = self.get_log_probs(self.policy, y_w)
        policy_logp_l = self.get_log_probs(self.policy, y_l)
        
        # get reference log probabilities
        with torch.no_grad():
            ref_logp_w = self.get_log_probs(self.reference, y_w)
            ref_logp_l = self.get_log_probs(self.reference, y_l)
        
        # compute log ratios
        log_ratio_w = policy_logp_w - ref_logp_w
        log_ratio_l = policy_logp_l - ref_logp_l
        
        # loss
        losses = -logsigmoid(self.beta * (log_ratio_w - log_ratio_l))
        return losses.mean()

    def update(self, y_w, y_l, x):
        """Single update step with preference data"""
        self.optimizer.zero_grad()
        loss = self.compute_loss(y_w, y_l, x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def generate_synthetic_data(num_samples=100, vocab_size=100, seq_length=5):
    """Generate synthetic preference data"""
    data = []
    for _ in range(num_samples):
        # random context (not used in this simplified example)
        x = torch.randint(0, vocab_size, (1,))
        
        # generate preferred completion (even tokens)
        y_w = torch.randint(0, vocab_size//2, (seq_length,)) * 2
        
        # generate dispreferred completion (odd tokens)
        y_l = torch.randint(0, vocab_size//2, (seq_length,)) * 2 + 1
        
        data.append((x, y_w, y_l))
    return data

def test_dpo():
    print("Testing DPO Agent...")
    
    # initialize agent and data
    agent = DPOAgent(vocab_size=100, beta=0.5)
    data = generate_synthetic_data(num_samples=100)
    
    # convert data to tensors
    x_batch = torch.stack([d[0] for d in data])
    y_w_batch = [d[1] for d in data]
    y_l_batch = [d[2] for d in data]
    
    # initial loss
    initial_loss = agent.compute_loss(y_w_batch, y_l_batch, x_batch)
    
    # perform update
    agent.update(y_w_batch, y_l_batch, x_batch)
    
    # check loss decrease
    new_loss = agent.compute_loss(y_w_batch, y_l_batch, x_batch)
    assert new_loss < initial_loss, "Loss should decrease after update"
    
    # check policy improvement
    with torch.no_grad():
        # get average probability for preferred tokens
        policy_probs_w = torch.exp(agent.get_log_probs(agent.policy, y_w_batch))
        policy_probs_l = torch.exp(agent.get_log_probs(agent.policy, y_l_batch))
        
    print(f"Initial Loss: {initial_loss:.4f}, New Loss: {new_loss:.4f}")
    print(f"Preferred Avg Prob: {policy_probs_w.mean():.4f}")
    print(f"Dispreferred Avg Prob: {policy_probs_l.mean():.4f}")
    assert policy_probs_w.mean() > policy_probs_l.mean(), "Policy should prefer y_w"
    print("DPO test passed!")

def train_dpo():
    # training setup
    agent = DPOAgent(vocab_size=1000, beta=0.1, lr=1e-4)
    data = generate_synthetic_data(num_samples=1000)
    
    # training loop
    print("Training DPO Agent...")
    for epoch in range(10):
        x_batch = torch.stack([d[0] for d in data])
        y_w_batch = [d[1] for d in data]
        y_l_batch = [d[2] for d in data]
        
        loss = agent.update(y_w_batch, y_l_batch, x_batch)
        
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                probs_w = torch.exp(agent.get_log_probs(agent.policy, y_w_batch)).mean()
                probs_l = torch.exp(agent.get_log_probs(agent.policy, y_l_batch)).mean()
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, "
                  f"P(y_w)={probs_w:.4f}, P(y_l)={probs_l:.4f}")

if __name__ == "__main__":
    test_dpo()
    train_dpo()