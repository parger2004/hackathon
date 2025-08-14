from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit
import torch  # Importing torch for tensor operations.
import numpy as np  # For numerical operations.

def encode_circuit_into_input_embedding(qc: QuantumCircuit, num_qubits: int, max_depth: int = 50) -> torch.Tensor:
    """
    Encode (convert) the quantum circuit into a tensor representation for input to the agent.
    
    Universal gate set encoding for VQE (RX, RY, RZ, H, CNOT):
    1. Circuit structure: which gates are applied where
    2. Parameter values: rotation angles for parameterized gates
    3. Circuit statistics: depth, gate counts, connectivity

    Args:
        qc (QuantumCircuit): The quantum circuit to be encoded.
        num_qubits (int): Total number of qubits in the system.
        max_depth (int): Maximum allowed circuit depth for padding.

    Returns:
        torch.Tensor: The tensor representation of the quantum circuit.
    """
    
    # Gate encoding: RX=0, RY=1, RZ=2, H=3, CNOT=4
    num_gate_types = 5
    
    # 1. Circuit structure matrix: [max_depth, num_qubits, num_gate_types]
    circuit_structure = torch.zeros(max_depth, num_qubits, num_gate_types)
    
    # 2. Parameter matrix: [max_depth, num_qubits] for rotation gates
    parameter_matrix = torch.zeros(max_depth, num_qubits)
    
    # 3. CNOT connectivity: [max_depth, num_qubits, num_qubits] 
    cnot_connections = torch.zeros(max_depth, num_qubits, num_qubits)
    
    # Process each instruction in the circuit
    depth_counter = 0
    for instruction in qc.data:
        if depth_counter >= max_depth:
            break
            
        gate = instruction.operation
        qubits = [qc.find_bit(qubit).index for qubit in instruction.qubits]
        gate_name = gate.name.lower()
        
        # Single qubit parameterized gates
        if gate_name == 'rx' and len(qubits) == 1:
            qubit_idx = qubits[0]
            circuit_structure[depth_counter, qubit_idx, 0] = 1.0  # RX gate
            
            if hasattr(gate, 'params') and len(gate.params) > 0:
                param_val = float(gate.params[0]) if gate.params[0] is not None else 0.0
                parameter_matrix[depth_counter, qubit_idx] = param_val / np.pi
                
        elif gate_name == 'ry' and len(qubits) == 1:
            qubit_idx = qubits[0]
            circuit_structure[depth_counter, qubit_idx, 1] = 1.0  # RY gate
            
            if hasattr(gate, 'params') and len(gate.params) > 0:
                param_val = float(gate.params[0]) if gate.params[0] is not None else 0.0
                parameter_matrix[depth_counter, qubit_idx] = param_val / np.pi
                
        elif gate_name == 'rz' and len(qubits) == 1:
            qubit_idx = qubits[0]
            circuit_structure[depth_counter, qubit_idx, 2] = 1.0  # RZ gate
            
            if hasattr(gate, 'params') and len(gate.params) > 0:
                param_val = float(gate.params[0]) if gate.params[0] is not None else 0.0
                parameter_matrix[depth_counter, qubit_idx] = param_val / np.pi
        
        # Hadamard gate
        elif gate_name == 'h' and len(qubits) == 1:
            qubit_idx = qubits[0]
            circuit_structure[depth_counter, qubit_idx, 3] = 1.0  # H gate
        
        # CNOT gate
        elif gate_name in ['cx', 'cnot'] and len(qubits) == 2:
            control_qubit, target_qubit = qubits[0], qubits[1]
            circuit_structure[depth_counter, control_qubit, 4] = 1.0  # CNOT control
            circuit_structure[depth_counter, target_qubit, 4] = 1.0   # CNOT target
            
            # Record CNOT connection
            cnot_connections[depth_counter, control_qubit, target_qubit] = 1.0
        
        depth_counter += 1
    
    # 4. Circuit statistics: [7] 
    current_depth = min(depth_counter, max_depth)
    total_gates = len(qc.data)
    
    # Count different gate types
    rx_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'rx')
    ry_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'ry')
    rz_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'rz')
    h_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'h')
    cnot_count = sum(1 for inst in qc.data if inst.operation.name.lower() in ['cx', 'cnot'])
    
    # Calculate entanglement measure from CNOT connections
    total_cnots = torch.sum(cnot_connections)
    max_possible_cnots = max_depth * num_qubits * (num_qubits - 1)
    entanglement_density = total_cnots / max_possible_cnots if max_possible_cnots > 0 else 0.0
    
    statistics = torch.tensor([
        current_depth / max_depth,           # Normalized depth
        total_gates / (max_depth * num_qubits), # Gate density
        rx_count / max(total_gates, 1),      # RX gate ratio
        ry_count / max(total_gates, 1),      # RY gate ratio
        rz_count / max(total_gates, 1),      # RZ gate ratio
        h_count / max(total_gates, 1),       # H gate ratio
        entanglement_density.item()          # Entanglement density
    ])
    
    # Flatten and concatenate all components
    circuit_structure_flat = circuit_structure.flatten()
    parameter_matrix_flat = parameter_matrix.flatten()
    cnot_connections_flat = cnot_connections.flatten()
    
    # Final encoding: concatenate all features
    encoding = torch.cat([
        circuit_structure_flat,
        parameter_matrix_flat, 
        cnot_connections_flat,
        statistics
    ])
    
    return encoding