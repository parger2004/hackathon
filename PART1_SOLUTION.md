# Part 1 Solution: Feature Encoding Scheme and Action Space for Quantum Circuit Optimization

## Overview

This solution implements a comprehensive feature encoding scheme and action space design for training reinforcement learning agents to optimize parameterized quantum circuits for VQE. The approach uses a **universal gate set** that is both theoretically complete and practically efficient for variational quantum algorithms.

## 1. Universal Gate Set for VQE

### Gate Selection Rationale

The implementation uses a **complete universal gate set** suitable for VQE:

| Gate | Type | Qubits | Parameters | Purpose |
|------|------|--------|------------|---------|
| **RX** | Rotation | 1 | θ | X-axis rotation for state preparation |
| **RY** | Rotation | 1 | θ | Y-axis rotation for state preparation |
| **RZ** | Rotation | 1 | θ | Z-axis rotation for phase control |
| **H** | Fixed | 1 | - | Superposition creation |
| **CNOT** | Entangling | 2 | - | Two-qubit entanglement |

**Why this set?**
- **Theoretical completeness**: Any quantum circuit can be approximated
- **VQE practical relevance**: Common in variational ansätze (UCCSD, Hardware Efficient, etc.)
- **Parameter efficiency**: Good balance of expressivity and trainability
- **Hardware compatibility**: Available on most quantum devices

## 2. Feature Encoding Scheme (State Representation)

### Architecture

The state representation captures comprehensive circuit information through four components:

#### Component 1: Circuit Structure Matrix
- **Shape**: `[max_depth, num_qubits, 5]`
- **Purpose**: One-hot encoding of gate types at each position
- **Encoding**: RX=0, RY=1, RZ=2, H=3, CNOT=4
- **Scalability**: O(depth × qubits)

#### Component 2: Parameter Matrix
- **Shape**: `[max_depth, num_qubits]`
- **Purpose**: Rotation angles for RX, RY, RZ gates
- **Encoding**: Normalized to [-1, 1] using param/π
- **Scalability**: O(depth × qubits)

#### Component 3: CNOT Connectivity Tensor
- **Shape**: `[max_depth, num_qubits, num_qubits]`
- **Purpose**: Tracks entanglement structure over time
- **Encoding**: Binary connectivity at each depth layer
- **Scalability**: O(depth × qubits²)

#### Component 4: Circuit Statistics Vector
- **Shape**: `[7]`
- **Components**:
  1. Normalized circuit depth
  2. Overall gate density
  3. RX gate ratio
  4. RY gate ratio
  5. RZ gate ratio
  6. H gate ratio
  7. Entanglement density
- **Scalability**: Constant

### Total Encoding Size

For a system with `n` qubits and maximum depth `d`:
```
Total Size = 5×d×n + d×n + d×n² + 7
           = d×n×(6+n) + 7
```

**Example for LiH molecule (12 qubits, depth 50)**:
- Circuit structure: 50 × 12 × 5 = 3,000
- Parameters: 50 × 12 = 600  
- CNOT connectivity: 50 × 12 × 12 = 7,200
- Statistics: 7
- **Total: 10,807 features**

## 3. Action Space Design

### Action Structure

Each action is a simple 3-element vector:

```python
action = [gate_type, qubit_index, parameter_value]
```

Where:
- **gate_type**: {0=RX, 1=RY, 2=RZ, 3=H, 4=CNOT_control, 5=CNOT_target}
- **qubit_index**: Target qubit [0, num_qubits-1]  
- **parameter_value**: Rotation angle [-1, 1] (ignored for H)

### CNOT Gate Handling

CNOT gates require two consecutive actions:
1. Action with gate_type=4 (control qubit)
2. Action with gate_type=5 (target qubit)

This allows the RL agent to learn optimal qubit pairings for entanglement.

### Action Space Bounds

```python
action_space = Box(
    low=[0, 0, -1.0], 
    high=[5, num_qubits-1, 1.0],
    dtype=float32
)
```

## 4. VQE Integration

### Environment Methods

#### `compute_reward(circuit)`
- Computes VQE energy using quantum circuit
- Reward = -|energy - FCI_energy| × 1000
- Bonus for convergence within tolerance (10⁻⁵)
- Penalty for excessive circuit depth

#### `step(action)`
- Parses action vector
- Applies gate to current circuit
- Computes reward via energy evaluation
- Updates state encoding
- Checks termination conditions

#### `reset()`
- Initializes with Hartree-Fock state
- Encodes initial circuit as state
- Resets episode counters

### Reward Structure

```python
if |energy - FCI_energy| < conv_tol:
    reward = 100.0  # Convergence bonus
else:
    reward = -|energy - FCI_energy| × 1000
    
reward -= 0.1 × circuit_depth  # Efficiency penalty
```

## 5. Implementation Validation

### Test Results

✅ **Universal Gate Coverage**: All 5 gate types correctly implemented  
✅ **Encoding Scalability**: Proper scaling from 4 to 16+ qubits  
✅ **VQE Ansatz Construction**: Can build typical variational circuits  
✅ **Action Space Validity**: 72 test actions all within bounds  
✅ **Circuit-State Consistency**: Different circuits produce distinct encodings  

### Performance Characteristics

- **State Size**: ~11K features for 12-qubit LiH
- **Action Space**: Continuous 3D space  
- **Memory**: O(depth × qubits²) for full encoding
- **Computation**: Linear in circuit size for encoding/decoding
- **Expressivity**: Universal quantum computation within depth limits

## 6. Advantages for VQE Training

1. **Complete Gate Vocabulary**: Full rotational and entangling control
2. **Efficient Representation**: Compact yet expressive state encoding
3. **Physical Relevance**: Gate set matches real VQE implementations
4. **Scalable Design**: Natural extension to larger molecules
5. **Convergence-Oriented**: Reward structure drives toward FCI accuracy
6. **Hardware-Ready**: Gates available on NISQ devices

## 7. Usage Example

```python
from env import VQEnv
import numpy as np

# Create VQE environment
env = VQEnv(qubit_operator=hamiltonian, 
           num_qubits=12, 
           fci_energy=-7.882,
           max_circuit_depth=50)

# Training loop
for episode in range(1000):
    state, info = env.reset()
    
    for step in range(20):
        # Agent chooses action: [gate_type, qubit, parameter]
        action = np.array([1, 0, 0.5])  # RY(π/2) on qubit 0
        
        next_state, reward, done, truncated, info = env.step(action)
        
        if done:  # Converged to FCI energy
            print(f"Converged in {step} steps!")
            break
            
        state = next_state
```

This implementation provides a solid foundation for training RL agents to discover optimal VQE ansätze that achieve chemical accuracy (10⁻⁵ Hartree) efficiently.
