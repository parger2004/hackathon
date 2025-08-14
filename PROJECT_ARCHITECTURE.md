# 🧬 Quantum Circuit Optimization with Reinforcement Learning
## Complete Project Architecture Guide

### 📋 **Project Overview**
This project uses **Reinforcement Learning (RL)** to automatically design quantum circuits for **Variational Quantum Eigensolver (VQE)** algorithms. Instead of manually designing quantum circuits for chemistry problems, an AI agent learns to build optimal circuits by trial and error.

**Goal**: Find the ground state energy of molecules (like LiH) by training an AI to construct quantum circuits that minimize energy.

---

## 🏗️ **Project Structure & File Connections**

```
src/
├── main.py                    # 🚀 Main orchestrator - starts everything
├── config_lih.cfg            # ⚙️ Configuration file - all parameters
├── env.py                    # 🎮 RL Environment - the "game" rules
├── agent.py                  # 🧠 PPO Agent - the AI brain
├── actor_critic_networks.py  # 🔗 Neural networks - decision making
├── memory.py                 # 💾 Experience buffer - stores learning data
├── helper_functions/
│   ├── encoding.py           # 📊 Circuit → Numbers converter
│   ├── decoding.py           # 🔧 Numbers → Circuit converter
│   ├── load_qubit_op.py      # 📂 Load molecular data
│   └── save_qubit_op.py      # 💿 Save molecular data
├── operators/
│   └── qubit_op_LiH.qpy     # 🧪 Pre-computed LiH molecule data
└── test_*.py                 # ✅ Validation and testing scripts
```

---

## 🔄 **How Components Connect: The Complete Data Flow**

### **1. INITIALIZATION PHASE**
```
main.py → config_lih.cfg → load_qubit_op.py → env.py → agent.py
```

### **2. TRAINING LOOP**
```
agent.py → env.py → encoding.py → agent.py → decoding.py → env.py → agent.py
    ↑                                                                      ↓
    └─────────────────── memory.py ←─────────────────────────────────────┘
```

### **3. QUANTUM SIMULATION**
```
decoding.py → QuantumCircuit → VQE → Energy → Reward → Learning
```

---

## 📁 **File-by-File Breakdown**

### 🚀 **main.py - The Central Orchestrator**
**Purpose**: Controls the entire training process and coordinates all components.

**What it does**:
1. **Loads configuration** from `config_lih.cfg`
2. **Loads molecular data** using `load_qubit_op.py`
3. **Creates environment** (`VQEnv` from `env.py`)
4. **Creates AI agent** (`PPOAgent` from `agent.py`)
5. **Runs training loop** (currently incomplete - has placeholder)

**Key variables from config**:
- `mol_name`: Molecule name ("LiH")
- `atoms`: Chemical symbols (['Li', 'H'])
- `coordinates`: 3D positions of atoms
- `fci_energy`: Target ground state energy (-7.88267...)
- `learning_rate`, `batch_size`, etc.: AI training parameters

**Current status**: ✅ Complete setup, ⚠️ Training loop needs implementation

**Connections**:
- **Reads**: `config_lih.cfg`, `operators/qubit_op_LiH.qpy`
- **Uses**: `env.py`, `agent.py`, `load_qubit_op.py`
- **Creates**: Environment and agent instances

---

### ⚙️ **config_lih.cfg - Configuration File**
**Purpose**: Stores all parameters in one place for easy modification.

**Sections**:
```ini
[MOL]  # Molecule properties
mol_name = LiH
atoms = ['Li', 'H']
coordinates = [([0.0, 0.0, 0.0], [1.6, 0.0, 0.0])]
fci_energy = -7.88266974664723
num_qubits = 12

[TRAIN]  # Training hyperparameters
learning_rate = 0.0003
num_episodes = 1000
max_circuit_depth = 50
conv_tol = 1e-5
```

**Why separate config**: Easy to experiment with different molecules and training settings without changing code.

---

### 🎮 **env.py - The RL Environment (The "Game")**
**Purpose**: Defines the rules of the quantum circuit optimization "game".

**Key components**:

#### **VQEnv Class**:
- **State space**: 10,807-dimensional vector (encoded quantum circuit)
- **Action space**: 3D continuous space `[gate_type, qubit_index, parameter]`
- **Reward function**: Energy-based scoring (closer to target = higher reward)

#### **Key methods**:
```python
__init__()     # Setup molecule, action/observation spaces
reset()        # Start new episode with Hartree-Fock initial state
step(action)   # Apply action, compute reward, return new state
compute_reward() # VQE energy calculation and reward assignment
```

#### **Universal Gate Set Definition**:
```python
# Action encoding: [gate_type, qubit_index, parameter_value]
# gate_type: 0=RX, 1=RY, 2=RZ, 3=H, 4=CNOT_control, 5=CNOT_target
self.action_space = gym.spaces.Box(
    low=np.array([0, 0, -1.0]), 
    high=np.array([5, num_qubits-1, 1.0])
)
```

**Connections**:
- **Uses**: `encoding.py` (state representation), `decoding.py` (action processing)
- **Provides**: Interface for agent to interact with quantum circuits
- **Integrates**: Qiskit quantum simulation with RL framework

---

### 🧠 **agent.py - The PPO Agent (AI Brain)**
**Purpose**: The learning algorithm that gets better at designing quantum circuits.

**PPO (Proximal Policy Optimization)**: Advanced RL algorithm that learns through experience.

#### **Key components**:
```python
class PPOAgent:
    def __init__()        # Setup networks, optimizers, hyperparameters
    def sample_action()   # ⚠️ NOT IMPLEMENTED - how to choose actions
    def store_transitions() # Save experience for learning
    def learn()           # ⚠️ NOT IMPLEMENTED - update neural networks
    def save_models()     # Save trained networks to disk
    def load_models()     # Load pre-trained networks
```

#### **Neural networks**:
- **Actor network**: Decides what actions to take
- **Critic network**: Evaluates how good situations are
- **Memory buffer**: Stores experiences for batch learning

**Current status**: ✅ Framework complete, ⚠️ Core methods need implementation

**Connections**:
- **Uses**: `actor_critic_networks.py`, `memory.py`
- **Receives**: States from `env.py`
- **Sends**: Actions to `env.py`

---

### 🔗 **actor_critic_networks.py - Neural Network Architectures**
**Purpose**: Defines the "brain structure" of the AI agent.

#### **ActorNetwork**:
```python
Input: [10,807 numbers] → Hidden layers → Output: [5 gate probabilities]
```
- **Job**: Look at current circuit, decide which gate to add next
- **Architecture**: `state_dim → 128 → 128 → num_gate_types`
- **Output**: Probability distribution over gate types

#### **CriticNetwork**:
```python
Input: [10,807 numbers] → Hidden layers → Output: [1 value score]
```
- **Job**: Evaluate how good/bad the current circuit state is
- **Architecture**: `state_dim → 256 → 256 → 1`
- **Output**: Single value (-∞ to +∞) rating the state

**Current status**: ✅ Basic structure implemented, ⚠️ Actor only chooses gate types (needs qubit selection and parameters)

**Connections**:
- **Used by**: `agent.py`
- **Processes**: Encoded states from `encoding.py`

---

### 💾 **memory.py - Experience Buffer**
**Purpose**: Stores agent experiences for batch learning.

**PPO Memory Buffer**: Collects multiple experiences before learning (more stable than learning from single experiences).

**Stores**:
- **States**: Circuit representations
- **Actions**: What the agent did
- **Rewards**: How good the results were
- **Values**: Critic network predictions
- **Probabilities**: Action likelihood

**Current status**: ⚠️ Implementation needed

**Connections**:
- **Used by**: `agent.py`
- **Stores**: Data from `env.py` interactions

---

### 📊 **encoding.py - Circuit → Numbers Converter**
**Purpose**: Translates quantum circuits into numerical vectors for neural networks.

#### **Encoding strategy** (3-part representation):
1. **Circuit structure**: `[50×12×5]` tensor - which gates where
2. **Parameters**: `[50×12]` matrix - rotation angles
3. **Statistics**: `[7]` vector - global circuit properties

#### **Universal gate recognition**:
```python
if gate_name == 'rx':  circuit_structure[layer, qubit, 0] = 1.0
if gate_name == 'ry':  circuit_structure[layer, qubit, 1] = 1.0
if gate_name == 'rz':  circuit_structure[layer, qubit, 2] = 1.0
if gate_name == 'h':   circuit_structure[layer, qubit, 3] = 1.0
if gate_name == 'cx':  circuit_structure[layer, qubit, 4] = 1.0
```

**Output**: Single tensor with **10,807 numbers** describing entire circuit.

**Connections**:
- **Called by**: `env.py` (in `reset()` and `step()`)
- **Input**: Qiskit `QuantumCircuit` objects
- **Output**: PyTorch tensors for neural networks

---

### 🔧 **decoding.py - Numbers → Circuit Converter**
**Purpose**: Converts AI decisions into actual quantum gate operations.

#### **Action processing**:
```python
def decode_actions_into_circuit(actions, num_qubits, base_circuit):
    for action in actions:
        gate_type, qubit_index, parameter_value = action
        
        if gate_type == 0:    # RX gate
            qc.rx(parameter_value * π, qubit_index)
        elif gate_type == 1:  # RY gate
            qc.ry(parameter_value * π, qubit_index)
        # ... etc for all gate types
```

#### **CNOT handling**:
- **Control action**: `gate_type = 4` specifies control qubit
- **Target action**: `gate_type = 5` specifies target qubit
- **Pairing**: Decoder looks for control+target pairs to build CNOT gates

**Connections**:
- **Called by**: `env.py` (in `step()` method)
- **Input**: Action tuples from agent
- **Output**: Modified `QuantumCircuit` objects

---

### 📂 **load_qubit_op.py / save_qubit_op.py - Data Persistence**
**Purpose**: Save/load pre-computed molecular Hamiltonian operators.

**Why needed**: Computing molecular Hamiltonians is expensive, so we pre-compute and save them.

#### **Workflow**:
1. **First time**: Use `save_qubit_op.py` to compute and save LiH Hamiltonian
2. **Training**: Use `load_qubit_op.py` to quickly load saved data
3. **Result**: Skip expensive quantum chemistry calculations during training

**File format**: `.qpy` (Qiskit's quantum program format)

**Connections**:
- **Used by**: `main.py`
- **Enables**: Fast environment initialization

---

### 🧪 **operators/qubit_op_LiH.qpy - Molecular Data**
**Purpose**: Pre-computed quantum representation of LiH molecule.

**Contents**:
- **Hamiltonian**: Mathematical description of molecular energy
- **12 qubits**: Quantum representation of molecular orbitals
- **Pauli operators**: X, Y, Z gate combinations for energy calculation

**Why LiH**: Simple molecule but complex enough to demonstrate VQE principles.

---

### ✅ **test_*.py - Validation Scripts**
**Purpose**: Verify that all components work correctly.

#### **test_universal.py**:
- Tests universal gate set implementation
- Validates encoding/decoding consistency
- Checks action space coverage

#### **test_part1.py**:
- Tests feature encoding scheme
- Validates action space design
- Comprehensive integration testing

#### **test_simplified.py**:
- Basic functionality tests
- Simplified examples for debugging

**Usage**: Run before training to ensure everything works.

---

## 🔄 **Complete Interaction Flow**

### **Training Episode Walkthrough**:

1. **Episode Start** (`main.py`):
   ```python
   observation = env.reset()  # Get initial state
   ```

2. **Environment Reset** (`env.py`):
   ```python
   self.current_circuit = HartreeFock_state  # Start with Hartree-Fock
   state = encoding.encode_circuit_into_input_embedding(circuit)
   return state  # 10,807 numbers describing initial circuit
   ```

3. **Agent Decision** (`agent.py`):
   ```python
   action = agent.sample_action(observation)  # Actor network chooses action
   # Returns: [gate_type, qubit_index, parameter] e.g., [1, 5, 0.7]
   ```

4. **Environment Step** (`env.py`):
   ```python
   new_circuit = decoding.decode_actions_into_circuit([action], circuit)
   energy = VQE_simulation(new_circuit)  # Quantum energy calculation
   reward = compute_reward(energy)  # Compare to target energy
   new_state = encoding.encode_circuit_into_input_embedding(new_circuit)
   return new_state, reward, done, info
   ```

5. **Learning** (`agent.py`):
   ```python
   agent.store_transitions(state, action, reward, ...)  # Save experience
   if memory_full:
       agent.learn()  # Update neural networks
   ```

6. **Repeat**: Steps 3-5 until episode ends (convergence or max steps)

---

## 🎯 **Key Design Decisions**

### **Universal Gate Set Choice**:
- **{RX, RY, RZ, H, CNOT}**: Theoretically complete and VQE-efficient
- **Alternative rejected**: {RY, CNOT} too minimal for complex molecules
- **Reasoning**: Balance between expressiveness and trainability

### **Encoding Strategy**:
- **Multi-component**: Structure + Parameters + Statistics
- **Fixed size**: 10,807 dimensions regardless of circuit complexity
- **Normalization**: All values in [-1, 1] range for neural network stability

### **Reward Function**:
- **Energy-based**: Closer to FCI energy = higher reward
- **Convergence bonus**: +100 for reaching target within tolerance
- **Depth penalty**: Encourage efficient circuits
- **Failure handling**: -1000 for invalid actions

---

## 🚧 **Current Implementation Status**

### ✅ **Completed Components**:
- Environment setup and configuration loading
- Universal gate set encoding/decoding
- VQE energy calculation and reward function
- Neural network architectures (basic)
- Molecular data loading/saving
- Comprehensive testing framework

### ⚠️ **Missing Implementation**:
- **Agent core methods**: `sample_action()` and `learn()` in `agent.py`
- **Memory buffer**: Complete PPO memory management in `memory.py`
- **Training loop**: Main training orchestration in `main.py`
- **Actor network**: Complete action generation (currently only gate types)

### 🎯 **Next Development Steps**:
1. Implement PPO agent core methods
2. Complete memory buffer functionality
3. Finish training loop in main.py
4. Add comprehensive logging and visualization
5. Benchmark against classical VQE approaches

---

## 🔬 **Scientific Context**

### **VQE (Variational Quantum Eigensolver)**:
- **Purpose**: Find ground state energies of molecules
- **Quantum advantage**: Exponential scaling for large molecules
- **Challenge**: Designing good ansatz circuits (solved by this project)

### **Reinforcement Learning Application**:
- **Traditional**: Humans design circuits based on chemical intuition
- **This approach**: AI discovers novel circuit architectures automatically
- **Potential impact**: Drug discovery, materials science, catalyst design

### **Technical Innovation**:
- **First**: RL for automated quantum circuit design
- **Scalable**: Works for arbitrary molecules and qubit counts
- **Practical**: Uses real quantum hardware constraints

---

## 🚀 **Getting Started Guide**

### **For New Contributors**:
1. **Start with tests**: Run `test_universal.py` to understand components
2. **Read config**: Examine `config_lih.cfg` to understand parameters
3. **Trace data flow**: Follow encoding → environment → decoding chain
4. **Implement missing pieces**: Start with `agent.py` core methods

### **For Researchers**:
1. **Modify molecules**: Change config to test different molecules
2. **Experiment with gates**: Extend universal gate set in encoding/decoding
3. **Tune rewards**: Adjust reward function for different optimization objectives
4. **Scale up**: Test with larger molecules and qubit counts

### **For Quantum Computing Students**:
1. **Understand VQE**: Learn how quantum circuits calculate molecular energies
2. **Study encoding**: See how classical ML interfaces with quantum algorithms
3. **Explore universality**: Experiment with different gate sets
4. **Run simulations**: Use Qiskit to see actual quantum circuit execution

This architecture enables automated discovery of quantum algorithms through machine learning - a powerful combination for solving real-world chemistry problems! 🧬⚛️🤖
