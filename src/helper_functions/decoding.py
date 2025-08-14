from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit # Importing QuantumCircuit for circuit operations.
import torch  # Importing torch for tensor operations.
import numpy as np  # For numerical operations.

def decode_actions_into_circuit(actions: List, num_qubits: int, base_circuit: QuantumCircuit = None) -> QuantumCircuit:
    """ 
    Decode the actions taken by the agent into quantum circuit operations.
    
    Universal Gate Set for VQE:
    - RX, RY, RZ: Single-qubit rotations (parameterized)
    - CNOT: Two-qubit entangling gate
    - H: Hadamard gate (superposition)
    
    Action Encoding:
    - Each action is a tuple: (gate_type, qubit_index, parameter_value)
    - gate_type: 0=RX, 1=RY, 2=RZ, 3=H, 4=CNOT(control), 5=CNOT(target)
    - qubit_index: target qubit for the gate
    - parameter_value: rotation angle for parameterized gates (ignored for H)

    Args:
        actions (List): A list of actions taken by the agent.
        num_qubits (int): Number of qubits in the circuit.
        base_circuit (QuantumCircuit): Optional base circuit to build upon.

    Returns:
        qc (QuantumCircuit): A quantum circuit constructed based on the agent's actions.
    """
    
    # Create quantum circuit
    if base_circuit is not None:
        qc = base_circuit.copy()
    else:
        qc = QuantumCircuit(num_qubits)
    
    # Process actions
    i = 0
    while i < len(actions):
        action = actions[i]
        
        if isinstance(action, (list, tuple)) and len(action) >= 3:
            gate_type, qubit_index, parameter_value = action[0], action[1], action[2]
            
            # Validate qubit index
            if not (0 <= qubit_index < num_qubits):
                i += 1
                continue
                
            try:
                if gate_type == 0:  # RX gate
                    angle = parameter_value * np.pi
                    qc.rx(angle, qubit_index)
                    
                elif gate_type == 1:  # RY gate
                    angle = parameter_value * np.pi
                    qc.ry(angle, qubit_index)
                    
                elif gate_type == 2:  # RZ gate
                    angle = parameter_value * np.pi
                    qc.rz(angle, qubit_index)
                    
                elif gate_type == 3:  # H gate
                    qc.h(qubit_index)
                    
                elif gate_type == 4:  # CNOT gate (control qubit specified)
                    # Look for next action to get target qubit
                    if i + 1 < len(actions):
                        next_action = actions[i + 1]
                        if (isinstance(next_action, (list, tuple)) and 
                            len(next_action) >= 3 and 
                            next_action[0] == 5):  # Target qubit action
                            
                            target_qubit = next_action[1]
                            if (0 <= target_qubit < num_qubits and 
                                target_qubit != qubit_index):
                                qc.cx(qubit_index, target_qubit)
                                i += 1  # Skip next action as it's consumed
                            
            except Exception:
                # Skip invalid gate applications
                pass
                
        i += 1
    
    return qc