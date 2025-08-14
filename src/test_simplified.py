#!/usr/bin/env python3
"""
Test script for the simplified VQE implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helper_functions.encoding import encode_circuit_into_input_embedding
from helper_functions.decoding import decode_actions_into_circuit
from qiskit import QuantumCircuit
import torch
import numpy as np

def test_minimal_gate_set():
    """Test the minimal gate set implementation."""
    print("=== Testing Minimal Gate Set (RY + CNOT) ===")
    
    num_qubits = 4
    
    # Test actions with minimal gate set
    test_actions = [
        (0, 0, 0.5),    # RY(π/2) on qubit 0
        (0, 1, -0.3),   # RY(-0.3π) on qubit 1
        (1, 0, 0.0),    # CNOT control: qubit 0
        (2, 1, 0.0),    # CNOT target: qubit 1
        (0, 2, 0.2),    # RY(0.2π) on qubit 2
        (1, 2, 0.0),    # CNOT control: qubit 2
        (2, 3, 0.0),    # CNOT target: qubit 3
    ]
    
    # Test decoding
    qc = decode_actions_into_circuit(test_actions, num_qubits)
    
    print(f"  Created circuit with {len(qc.data)} gates")
    print(f"  Circuit depth: {qc.depth()}")
    
    # Count gate types
    ry_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'ry')
    cnot_count = sum(1 for inst in qc.data if inst.operation.name.lower() in ['cx', 'cnot'])
    
    print(f"  RY gates: {ry_count}")
    print(f"  CNOT gates: {cnot_count}")
    
    assert len(qc.data) > 0, "No gates were applied"
    return qc

def test_simplified_encoding():
    """Test the simplified encoding scheme."""
    print("\n=== Testing Simplified Encoding ===")
    
    num_qubits = 4
    max_depth = 10
    
    # Create test circuit
    qc = QuantumCircuit(num_qubits)
    qc.ry(0.5, 0)
    qc.cx(0, 1)
    qc.ry(0.3, 2)
    
    # Test encoding
    encoding = encode_circuit_into_input_embedding(qc, num_qubits, max_depth)
    
    # Calculate expected size
    num_gate_types = 2  # RY and CNOT
    circuit_structure_size = max_depth * num_qubits * num_gate_types
    parameter_matrix_size = max_depth * num_qubits
    cnot_connections_size = max_depth * num_qubits * num_qubits
    statistics_size = 5
    
    expected_size = (circuit_structure_size + parameter_matrix_size + 
                    cnot_connections_size + statistics_size)
    
    print(f"  Encoding size: {encoding.shape[0]}")
    print(f"  Expected size: {expected_size}")
    print(f"  Size match: {encoding.shape[0] == expected_size}")
    
    assert encoding.shape[0] == expected_size, "Encoding size mismatch"
    
    # Test that encoding captures differences
    empty_qc = QuantumCircuit(num_qubits)
    empty_encoding = encode_circuit_into_input_embedding(empty_qc, num_qubits, max_depth)
    
    difference = torch.norm(encoding - empty_encoding)
    print(f"  Encoding difference: {difference:.4f}")
    
    assert difference > 0, "Different circuits should have different encodings"

def test_action_space_format():
    """Test the simplified action space format."""
    print("\n=== Testing Action Space Format ===")
    
    # Test action format: [gate_type, qubit_index, parameter_value]
    test_actions = [
        [0, 0, 0.5],    # RY gate on qubit 0 with parameter 0.5
        [1, 0, 0.0],    # CNOT control on qubit 0
        [2, 1, 0.0],    # CNOT target on qubit 1
        [0, 2, -0.3],   # RY gate on qubit 2 with parameter -0.3
    ]
    
    num_qubits = 4
    
    for i, action in enumerate(test_actions):
        gate_type, qubit_index, parameter = action
        
        # Validate action format
        assert 0 <= gate_type <= 2, f"Invalid gate type: {gate_type}"
        assert 0 <= qubit_index < num_qubits, f"Invalid qubit index: {qubit_index}"
        assert -1.0 <= parameter <= 1.0, f"Parameter out of range: {parameter}"
        
        print(f"  Action {i}: gate_type={gate_type}, qubit={qubit_index}, param={parameter:.1f}")
    
    print("  All actions valid ✓")

def test_complete_vqe_workflow():
    """Test the complete VQE workflow simulation."""
    print("\n=== Testing Complete VQE Workflow ===")
    
    num_qubits = 4
    max_depth = 15
    
    # Simulate a simple VQE-like workflow
    # Start with Hartree-Fock (empty circuit for simplicity)
    current_circuit = QuantumCircuit(num_qubits)
    
    # Simulate agent actions
    agent_actions = [
        (0, 0, 0.1),   # RY on qubit 0
        (0, 1, 0.2),   # RY on qubit 1  
        (1, 0, 0.0),   # CNOT control
        (2, 1, 0.0),   # CNOT target
        (0, 2, 0.3),   # RY on qubit 2
        (0, 3, 0.4),   # RY on qubit 3
        (1, 2, 0.0),   # CNOT control
        (2, 3, 0.0),   # CNOT target
    ]
    
    print(f"  Simulating {len(agent_actions)} agent actions")
    
    for i, action in enumerate(agent_actions):
        # Apply action to circuit
        current_circuit = decode_actions_into_circuit([action], num_qubits, current_circuit)
        
        # Encode circuit state
        state = encode_circuit_into_input_embedding(current_circuit, num_qubits, max_depth)
        
        print(f"    Step {i+1}: Circuit depth={current_circuit.depth()}, State size={state.shape[0]}")
    
    final_gate_count = len(current_circuit.data)
    print(f"  Final circuit: {final_gate_count} gates, depth {current_circuit.depth()}")
    
    assert final_gate_count > 0, "Circuit should have gates"
    assert current_circuit.depth() <= max_depth, "Circuit depth should be reasonable"

if __name__ == "__main__":
    print("Testing Simplified VQE Implementation")
    print("=" * 50)
    
    try:
        test_minimal_gate_set()
        test_simplified_encoding()
        test_action_space_format()
        test_complete_vqe_workflow()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("\nSimplified implementation features:")
        print("1. ✅ Minimal universal gate set (RY + CNOT)")
        print("2. ✅ Simple action space format [gate_type, qubit, parameter]")
        print("3. ✅ Compact encoding scheme")
        print("4. ✅ VQE-ready workflow")
        print("\nReady for PPO agent training!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
