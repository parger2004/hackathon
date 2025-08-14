# Importing libraries: 
import os # For file system access.
import torch # For tensor operations.
import torch.nn.functional as F # For loss functions.
from torch.optim import Adam, SGD # For gradient update.
from typing import Tuple # For typing annotation.

# Importing custom modules:
from actor_critic_networks import ActorNetwork, CriticNetwork
from memory import PPOMemory

class PPOAgent:
    '''
    Class for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    learning_rate: float
        Learning rate for the optimizer.
    gamma: float
        Discount factor for future rewards.
    gae_lambda: float
        Lambda parameter for Generalized Advantage Estimation (GAE).
    policy_clip: float
        Clipping parameter for the policy loss.
    batch_size: int
        Batch size for training.
    num_epochs: int
        Number of epochs for training.
    optimizer_option: str
        Choice of optimizer ('Adam' or 'SGD').
    chkpt_dir: str
        Directory to save the model checkpoint.
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 batch_size=64,
                 num_epochs=10,
                 optimizer_option="Adam",
                 chkpt_dir='model/ppo'):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Instantiate the Actor and Critic networks:
        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim=state_dim).to(self.device)

        # Define a dictionary for optimizers:
        optimizers = {
            "Adam": Adam,
            "SGD": SGD
            }
        OptimizerClass = optimizers.get(optimizer_option, Adam) # Default to Adam if not found.
        
        # Define optimizers:
        self.actor_optimizer = OptimizerClass(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = OptimizerClass(self.critic.parameters(), lr=learning_rate)

        # Buffer for storing transitions:
        self.memory_buffer = PPOMemory(batch_size)

        # Checkpoint paths:
        self.actor_chkpt = os.path.join(chkpt_dir, 'actor_net_torch_ppo')
        self.critic_chkpt = os.path.join(chkpt_dir, 'critic_net_torch_ppo')
        
    def sample_action(self, observation: torch.tensor) -> Tuple[list, list, float]:
        """
        Sample actions from the policy network given the current state (observation).

        Args:
            observation (torch.tensor): the state representation.

        Returns:
            action (list): list of action(s) [gate_type, qubit_index, parameter].
            probs (list): list of probability distribution(s) over action(s).
            value (float): the value from the Critic network.
        """
        
        # Set networks to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Move observation to device
            observation = observation.to(self.device)
            
            # Get action distributions from actor network
            gate_dist, qubit_dist, param_dist = self.actor(observation)
            
            # Sample actions from distributions
            gate_action = gate_dist.sample()
            qubit_action = qubit_dist.sample()
            param_action = param_dist.sample()
            
            # Clamp parameter to valid range [-1, 1]
            param_action = torch.clamp(param_action, -1.0, 1.0)
            
            # Get action probabilities (log probabilities)
            gate_log_prob = gate_dist.log_prob(gate_action)
            qubit_log_prob = qubit_dist.log_prob(qubit_action)
            param_log_prob = param_dist.log_prob(param_action)
            
            # Combined log probability
            total_log_prob = gate_log_prob + qubit_log_prob + param_log_prob
            
            # Get state value from critic
            value = self.critic(observation)
            
            # Convert to lists for environment
            action = [
                gate_action.item(),
                qubit_action.item(), 
                param_action.item()
            ]
            
            probs = [total_log_prob.exp().item()]  # Convert log prob to probability
            
        return action, probs, value.item()

    def store_transitions(self, state, action, reward, probs, vals, done):
        """
        This method stores transitions in the memory buffer.
        """
        self.memory_buffer.store_memory(state, action, reward, probs, vals, done)

    def learn(self):
        """
        PPO learning step - update actor and critic networks.
        """
        
        # Get all stored experiences
        states, actions, rewards, old_probs, values, dones = self.memory_buffer.generate_batches()
        
        # Compute GAE advantages and returns
        advantages, returns = self.memory_buffer.compute_gae_returns(
            gamma=self.gamma, 
            gae_lambda=self.gae_lambda
        )
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # PPO update for multiple epochs
        for epoch in range(self.num_epochs):
            # Set networks to training mode
            self.actor.train()
            self.critic.train()
            
            # Forward pass through actor
            gate_dist, qubit_dist, param_dist = self.actor(states)
            
            # Calculate log probabilities for taken actions
            gate_actions = actions[:, 0].long()
            qubit_actions = actions[:, 1].long()
            param_actions = actions[:, 2]
            
            gate_log_probs = gate_dist.log_prob(gate_actions)
            qubit_log_probs = qubit_dist.log_prob(qubit_actions)
            param_log_probs = param_dist.log_prob(param_actions)
            
            # Combined log probabilities
            new_log_probs = gate_log_probs + qubit_log_probs + param_log_probs
            
            # Calculate probability ratios
            ratio = torch.exp(new_log_probs - torch.log(old_probs + 1e-8))
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages
            
            # Actor loss (PPO objective)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus for exploration
            entropy = gate_dist.entropy() + qubit_dist.entropy() + param_dist.entropy()
            entropy_loss = -0.05 * entropy.mean()  # INCREASED entropy coefficient for better exploration
            
            # Total actor loss
            total_actor_loss = actor_loss + entropy_loss
            
            # Critic loss (value function)
            new_values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(new_values, returns)
            
            # Update actor network
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear memory after learning
        self.memory_buffer.clear_memory()
        
        # Print learning statistics
        print(f"    Actor Loss: {total_actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        print(f"    Entropy: {entropy.mean().item():.4f}, Advantage Mean: {advantages.mean().item():.4f}")

    def save_models(self):
        print("Saving models...")
        torch.save(self.actor.state_dict(), self.actor_chkpt)
        torch.save(self.critic.state_dict(), self.critic_chkpt)

    def load_models(self):
        print("Loading models...")
        self.actor.load_state_dict(torch.load(self.actor_chkpt))
        self.critic.load_state_dict(torch.load(self.critic_chkpt))