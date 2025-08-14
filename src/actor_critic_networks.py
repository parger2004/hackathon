import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class ActorNetwork(nn.Module):
    """
    Actor network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    """
    def __init__(self, state_dim, action_dim):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        # Store dimensions for action space
        self.num_qubits = int(action_dim.high[1]) + 1  # Extract num_qubits from action space
        
        # Shared feature extraction layers
        self.shared_fc1 = nn.Linear(state_dim, 256)
        self.shared_fc2 = nn.Linear(256, 128)
        
        # Gate type selection (categorical: 0-5)
        self.gate_type_head = nn.Linear(128, 6)  # RX, RY, RZ, H, CNOT_control, CNOT_target
        
        # Qubit selection (categorical: 0 to num_qubits-1)
        self.qubit_head = nn.Linear(128, self.num_qubits)
        
        # Parameter value (continuous: -1 to 1)
        self.parameter_head = nn.Linear(128, 1)

    def forward(self, state):
        """
        Forward pass to generate action distributions.
        
        Returns:
            gate_dist: Categorical distribution for gate types
            qubit_dist: Categorical distribution for qubit selection  
            parameter_dist: Normal distribution for parameter values
        """
        
        # Shared feature extraction
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Gate type distribution (categorical)
        gate_logits = self.gate_type_head(x)
        gate_dist = dist.Categorical(logits=gate_logits)
        
        # Qubit selection distribution (categorical)
        qubit_logits = self.qubit_head(x)
        qubit_dist = dist.Categorical(logits=qubit_logits)
        
        # Parameter distribution (continuous, constrained to [-1, 1])
        parameter_mean = torch.tanh(self.parameter_head(x))  # Constrain to [-1, 1]
        parameter_std = torch.ones_like(parameter_mean) * 0.3  # Fixed std for exploration
        parameter_dist = dist.Normal(parameter_mean, parameter_std)
        
        return gate_dist, qubit_dist, parameter_dist

class CriticNetwork(nn.Module):
    """
    Critic network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    fc1_dims: int
        Number of neurons in the first hidden layer.
    fc2_dims: int
        Number of neurons in the second hidden layer.
    """
    def __init__(self, state_dim, fc1_dims=256, fc2_dims=256):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        # Neural network layers:
        self.fcnn = nn.Sequential(
                nn.Linear(state_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

    def forward(self, state):
        """
        Forward pass.
        """
        value = self.fcnn(state)
        return value
