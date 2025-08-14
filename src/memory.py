import numpy as np
import torch

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, reward, probs, vals, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        """
        Generate batches of experiences for training.
        Returns all stored experiences (not batched for simplicity).
        """
        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.rewards),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.dones)

    def compute_gae_returns(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) returns and advantages.
        
        Args:
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            
        Returns:
            advantages (np.array): Computed advantages
            returns (np.array): Computed returns
        """
        rewards = np.array(self.rewards)
        values = np.array(self.vals)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            # Temporal difference error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE calculation
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Returns are advantages + values
        returns = advantages + values
        
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def __len__(self):
        """Return the number of stored experiences."""
        return len(self.states)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []