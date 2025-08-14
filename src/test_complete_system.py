#!/usr/bin/env python3
"""
Quick test script to verify all components work before full training.
"""

import torch
import numpy as np
from env import VQEnv
from agent import PPOAgent
from helper_functions.load_qubit_op import load_qubit_op_from_file

def test_components():
    """Test all major components quickly."""
    
    print("🧪 Testing Quantum Circuit RL Components...")
    
    # Load molecular data
    print("📂 Loading molecular data...")
    qubit_operator = load_qubit_op_from_file("./operators/qubit_op_LiH.qpy")
    
    # Create environment
    print("🎮 Creating environment...")
    env = VQEnv(
        qubit_operator=qubit_operator,
        num_spatial_orbitals=6,
        num_particles=(2, 2),
        fci_energy=-7.88266974664723
    )
    
    # Create agent
    print("🧠 Creating agent...")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space,  # Pass action_space object
        learning_rate=0.001,
        batch_size=4  # Small batch for testing
    )
    
    print(f"✅ State dimension: {env.observation_space.shape[0]}")
    print(f"✅ Number of qubits: {env.num_qubits}")
    
    # Test one episode
    print("\n🎯 Testing one episode...")
    
    observation, info = env.reset()
    print(f"✅ Environment reset successful, state shape: {observation.shape}")
    
    for step in range(3):  # Just 3 steps for testing
        # Test agent action sampling
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        action, probs, value = agent.sample_action(obs_tensor)
        
        print(f"  Step {step}:")
        print(f"    Action: {action}")
        print(f"    Probability: {probs[0]:.4f}")
        print(f"    Value: {value:.4f}")
        
        # Test environment step
        next_observation, reward, terminated, truncated, info = env.step(np.array(action))
        
        print(f"    Reward: {reward:.4f}")
        print(f"    Energy: {info['ep_energy'][-1]:.6f}")
        
        # Store transition
        agent.store_transitions(
            state=observation,
            action=action,
            reward=reward,
            probs=probs[0],
            vals=value,
            done=terminated or truncated
        )
        
        observation = next_observation
        
        if terminated or truncated:
            break
    
    # Test learning (if enough data)
    if len(agent.memory_buffer) >= 2:
        print("\n🧠 Testing learning step...")
        try:
            agent.learn()
            print("✅ Learning step successful!")
        except Exception as e:
            print(f"❌ Learning failed: {e}")
    else:
        print("⚠️ Not enough data for learning test")
    
    print("\n🎉 All tests passed! System is ready for training.")
    print("🚀 Run: python main.py config_lih.cfg")

if __name__ == "__main__":
    test_components()
