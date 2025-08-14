import configparser # Config file.
import numpy as np # Numerical operations.
import torch # PyTorch for tensors.
import ast # Convert string to list.
import sys # Command-line arguments.
import os # Directories.
import signal # For graceful interrupt handling.
import time # For timing operations.

# Helper functions:
#from helper_functions.save_qubit_op import save_qubit_op_to_file
from src.helper_functions.load_qubit_op import load_qubit_op_from_file

# Import the agent and environment classes:
from src.agent import PPOAgent
from src.env import VQEnv

##########################################
if __name__ == '__main__':
    # Parse command-line arguments:
    config_file = sys.argv[1]

    # Get the path to the config.cfg file:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, config_file)

    # Load the configuration file:
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Molecule hyperparameters:
    mol_name = config['MOL'].get('mol_name', fallback='Unknown')

    # Atoms:
    atoms_str = config['MOL'].get('atoms', fallback=None)
    atoms = ast.literal_eval(atoms_str) if atoms_str else []

    # Coordinates:
    coordinates_str = config['MOL'].get('coordinates', fallback=None)
    coordinates = ast.literal_eval(coordinates_str) if coordinates_str else ()

    # Number of particles:
    num_particles_str = config['MOL'].get('num_particles', fallback = None)
    num_particles = ast.literal_eval(num_particles_str) if num_particles_str else (0, 0)

    # Multiplicity:
    multiplicity = config.getint('MOL', 'multiplicity', fallback=1)
    # Charge:
    charge = config.getint('MOL', 'charge', fallback=0)
    # Electrons:
    num_electrons = config.getint('MOL', 'num_electrons', fallback = None)
    # Spatial orbitals:
    num_spatial_orbitals = config.getint('MOL', 'num_spatial_orbitals', fallback = None) 
    # Number of qubits:
    num_qubits = config.getint('MOL', 'num_qubits', fallback = None)
    # FCI energy:
    fci_energy = config.getfloat('MOL', 'fci_energy', fallback = None)

    # Convergence tolerance:
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)

    # Training hyperparameters:
    learning_rate = config.getfloat('TRAIN', 'learning_rate', fallback=0.0003)
    gamma = config.getfloat('TRAIN', 'gamma', fallback=0.99) 
    gae_lambda = config.getfloat('TRAIN', 'gae_lambda', fallback=0.95) 
    policy_clip = config.getfloat('TRAIN', 'policy_clip', fallback=0.2) 
    batch_size = config.getint('TRAIN', 'batch_size', fallback=64) 
    num_episodes = config.getint('TRAIN', 'num_episodes', fallback=100) # This is the number of episodes to train the agent.
    num_steps = config.getint('TRAIN', 'num_steps', fallback=20) # This is the number of steps per episode.
    num_epochs = config.getint('TRAIN', 'num_epochs', fallback=10) # This is the number of passes over the same batch of collected data for policy update.
    max_circuit_depth = config.getint('TRAIN', 'max_circuit_depth', fallback=50) 
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)
    optimizer_option = config['TRAIN'].get('optimizer_option', fallback='SGD')

    ##########################################

    '''
    # Create an instance of the VQEnv class:
    env = VQEnv(molecule_name = "LiH", 
                symbols = atoms, 
                geometry = coordinates, 
                multiplicity = multiplicity, 
                charge = charge,
                num_electrons = num_electrons,
                num_spatial_orbitals = num_spatial_orbitals)

    # Save the qubit operator to disk:
    save_qubit_op_to_file(qubit_op = env.qubit_operator, file_name = "qubit_op_LiH.qpy")
    '''
    
    # Load the qubit operator from disk:
    qubit_operator = load_qubit_op_from_file(file_path = "./src/operators/qubit_op_LiH.qpy")

    ##########################################

    # Create the environment with the loaded qubit operator:
    env = VQEnv(qubit_operator = qubit_operator, 
                num_spatial_orbitals = num_spatial_orbitals, 
                num_particles = num_particles,
                fci_energy = fci_energy)

    # Agent:
    agent = PPOAgent(
        state_dim = env.observation_space.shape[0],
        action_dim = env.action_space,  # Pass action_space object, not num_qubits
        learning_rate = learning_rate,
        gamma = gamma,
        gae_lambda = gae_lambda,
        policy_clip =policy_clip,
        batch_size = batch_size,
        num_epochs = num_epochs,
        optimizer_option = optimizer_option,
        chkpt_dir = 'model/ppo')

    # Training loop:
    print(f"\n🚀 Starting training for {num_episodes} episodes...")
    print("💡 Press Ctrl+C to safely interrupt training and save progress")
    
    # Variables to track training progress
    training_interrupted = False
    start_time = time.time()
    best_energy = float('inf')
    best_episode = 0
    
    try:
        for episode in range(num_episodes):
            # Reset environment for new episode
            observation, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            # Run episode
            for step in range(num_steps):
                # Convert observation to tensor
                obs_tensor = torch.tensor(observation, dtype=torch.float32)
                
                # Sample action from agent
                action, action_probs, value = agent.sample_action(obs_tensor)
                
                # Take step in environment
                next_observation, reward, terminated, truncated, info = env.step(np.array(action))
                
                # Store transition in memory
                agent.store_transitions(
                    state=observation,
                    action=action, 
                    reward=reward,
                    probs=action_probs[0],  # Extract single probability
                    vals=value,
                    done=terminated or truncated
                )
                
                # Update tracking
                episode_reward += reward
                episode_steps += 1
                observation = next_observation
                
                # Print step info
                if step % 5 == 0 or terminated or truncated:
                    print(f"  Step {step}: Action={action}, Reward={reward:.3f}, Energy={info['ep_energy'][-1]:.6f}")
                
                # End episode if terminated or truncated
                if terminated or truncated:
                    if terminated:
                        print(f"  ✅ Episode converged after {episode_steps} steps!")
                    else:
                        print(f"  ⏱️ Episode truncated after {episode_steps} steps")
                    break
            
            # Learn from collected experience
            if len(agent.memory_buffer.states) >= agent.memory_buffer.batch_size:
                print("  🧠 Learning from experience...")
                agent.learn()
            
            # Episode summary
            final_energy = info['ep_energy'][-1] if info['ep_energy'] else 0
            energy_error = abs(final_energy - fci_energy)
            
            # Track best result
            if energy_error < abs(best_energy - fci_energy):
                best_energy = final_energy
                best_episode = episode + 1
                print("  🌟 New best energy achieved!")
                
            print(f"  📈 Episode {episode + 1} Summary:")
            print(f"     Total Reward: {episode_reward:.3f}")
            print(f"     Final Energy: {final_energy:.6f}")
            print(f"     Target Energy: {fci_energy:.6f}")
            print(f"     Energy Error: {energy_error:.6f}")
            print(f"     Best Energy: {best_energy:.6f} (Episode {best_episode})")
            print(f"     Steps: {episode_steps}")
            
            # Save models periodically
            if (episode + 1) % 50 == 0:
                print(f"  💾 Saving models at episode {episode + 1}")
                agent.save_models()
            
            # Early stopping if converged
            if terminated and energy_error < conv_tol:
                print(f"\n🎉 Training converged! Target energy reached in episode {episode + 1}")
                break
                
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user at episode {episode + 1}")
        print("🔄 Performing graceful shutdown...")
        training_interrupted = True
        
        # Save current progress
        print("💾 Saving current model state...")
        agent.save_models()
        
        # Final summary of interrupted training
        elapsed_time = time.time() - start_time
        print(f"\n📊 Training Interrupted Summary:")
        print(f"   Episodes completed: {episode + 1}/{num_episodes}")
        print(f"   Training time: {elapsed_time:.2f} seconds")
        print(f"   Best energy achieved: {best_energy:.6f} (Episode {best_episode})")
        print(f"   Target energy: {fci_energy:.6f}")
        print(f"   Best error: {abs(best_energy - fci_energy):.6f}")
        
        # 🎯 DISPLAY FINAL BEST CIRCUIT (for interrupted training)
        print(f"\n🏆 FINAL BEST CIRCUIT VISUALIZATION:")
        print("=" * 80)
        try:
            print(f"📋 Best Result Summary:")
            print(f"   Episode: {best_episode}")
            print(f"   Energy: {best_energy:.6f}")
            print(f"   Target: {fci_energy:.6f}")
            print(f"   Error: {abs(best_energy - fci_energy):.6f}")
            print(f"   Convergence: {'✅ YES' if abs(best_energy - fci_energy) < conv_tol else '❌ NO'}")
            print("=" * 80)
            
            # Display current circuit (last episode's result)
            print(f"🔧 Final Circuit:")
            print(env.current_circuit.draw(output='text', fold=-1))
            print("=" * 80)
            
            print(f"📊 Final Circuit Statistics:")
            print(f"   Total Gates: {len(env.current_circuit.data)}")
            print(f"   Circuit Depth: {env.current_circuit.depth()}")
            print(f"   Qubits Used: {env.current_circuit.num_qubits}")
            
            # List all gates
            print(f"\n📝 Gate-by-Gate Breakdown:")
            for i, instruction in enumerate(env.current_circuit.data):
                gate_name = instruction.operation.name
                qubits = [env.current_circuit.find_bit(qubit).index for qubit in instruction.qubits]
                params = getattr(instruction.operation, 'params', [])
                param_str = f", θ={params[0]:.3f}" if params else ""
                print(f"   Gate {i+1}: {gate_name.upper()} on qubit(s) {qubits}{param_str}")
                
        except Exception as e:
            print(f"❌ Could not display final circuit: {e}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Attempting to save current progress...")
        try:
            agent.save_models()
            print("✅ Progress saved successfully")
        except:
            print("❌ Failed to save progress")
        raise e
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    if not training_interrupted:
        print(f"\n✅ Training completed successfully!")
        final_energy = info['ep_energy'][-1] if info['ep_energy'] else 0
        energy_error = abs(final_energy - fci_energy)
        
        print(f"📊 Final Results:")
        print(f"   Episodes: {episode + 1}")
        print(f"   Training time: {elapsed_time:.2f} seconds")
        print(f"   Final Energy: {final_energy:.6f}")
        print(f"   Target Energy: {fci_energy:.6f}")
        print(f"   Final Error: {energy_error:.6f}")
        print(f"   Best Energy: {best_energy:.6f} (Episode {best_episode})")
        
        # Save final models
        agent.save_models()
        print("💾 Final models saved!")
        
        # 🎯 DISPLAY FINAL BEST CIRCUIT
        print(f"\n🏆 FINAL BEST CIRCUIT VISUALIZATION:")
        print("=" * 80)
        try:
            # Find the best circuit from training
            print(f"📋 Best Result Summary:")
            print(f"   Episode: {best_episode}")
            print(f"   Energy: {best_energy:.6f}")
            print(f"   Target: {fci_energy:.6f}")
            print(f"   Error: {abs(best_energy - fci_energy):.6f}")
            print(f"   Convergence: {'✅ YES' if abs(best_energy - fci_energy) < conv_tol else '❌ NO'}")
            print("=" * 80)
            
            # Display current circuit (last episode's result)
            print(f"🔧 Current Circuit:")
            print(env.current_circuit.draw(output='text', fold=-1))
            print("=" * 80)
            
            print(f"📊 Final Circuit Statistics:")
            print(f"   Total Gates: {len(env.current_circuit.data)}")
            print(f"   Circuit Depth: {env.current_circuit.depth()}")
            print(f"   Qubits Used: {env.current_circuit.num_qubits}")
            
            # List all gates
            print(f"\n📝 Gate-by-Gate Breakdown:")
            for i, instruction in enumerate(env.current_circuit.data):
                gate_name = instruction.operation.name
                qubits = [env.current_circuit.find_bit(qubit).index for qubit in instruction.qubits]
                params = getattr(instruction.operation, 'params', [])
                param_str = f", θ={params[0]:.3f}" if params else ""
                print(f"   Gate {i+1}: {gate_name.upper()} on qubit(s) {qubits}{param_str}")
                
        except Exception as e:
            print(f"❌ Could not display final circuit: {e}")
        print("=" * 80)
    
    print(f"\n🏁 Training session ended.")
    print(f"📁 Model checkpoints saved to: model/ppo/")
    print(f"🔄 You can resume training by running the script again.")