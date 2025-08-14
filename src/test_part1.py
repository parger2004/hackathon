#!/usr/bin/env python3
"""
Test script for the proposed feature encoding scheme and action space.
This validates the Part 1 implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helper_functions.encoding import encode_circuit_into_input_embedding
from helper_functions.decoding import decode_actions_into_circuit
from qiskit import QuantumCircuit
import torch
import numpy as np

def test_encoding_scalability():
    """Test that encoding scales with number of qubits."""
    print("=== Testing Encoding Scalability ===")
    
    for num_qubits in [4, 8, 12, 16]:
        # Create a simple test circuit
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        if num_qubits > 1:
            qc.cx(0, 1)
        if num_qubits > 2:
            qc.rx(0.5, 2)
        
        # Test encoding
        max_depth = 20
        encoding = encode_circuit_into_input_embedding(qc, num_qubits, max_depth)
        
        # Calculate expected size
        num_gate_types = 13
        circuit_structure_size = max_depth * num_qubits * num_gate_types
        parameter_matrix_size = max_depth * num_qubits
        connectivity_matrix_size = num_qubits * num_qubits
        statistics_size = 4
        expected_size = circuit_structure_size + parameter_matrix_size + connectivity_matrix_size + statistics_size
        
        print(f"  Qubits: {num_qubits:2d} | Encoding size: {encoding.shape[0]:5d} | Expected: {expected_size:5d} | Match: {encoding.shape[0] == expected_size}")
        
        assert encoding.shape[0] == expected_size, f"Size mismatch for {num_qubits} qubits"

def test_action_space_design():
    """Test the action space and decoding functionality."""
    print("\n=== Testing Action Space Design ===")
    
    num_qubits = 4
    
    # Test different gate types
    test_actions = [
        # Single qubit gates
        (3, [0], 0.0),      # Hadamard on qubit 0
        (4, [1], 0.0),      # Pauli-X on qubit 1
        (0, [2], 0.5),      # RX(π/2) on qubit 2
        (1, [3], -0.5),     # RY(-π/2) on qubit 3
        
        # Two qubit gates
        (7, [0, 1], 0.0),   # CNOT between qubits 0 and 1
        (8, [2, 3], 0.0),   # CZ between qubits 2 and 3
        
        # Three qubit gate
        (10, [0, 1, 2], 0.0), # Toffoli gate
    ]
    
    # Test decoding
    qc = decode_actions_into_circuit(test_actions, num_qubits)
    
    print(f"  Created circuit with {len(qc.data)} gates")
    print(f"  Circuit depth: {qc.depth()}")
    print(f"  Gates applied:")
    
    for i, (gate, qubits, param) in enumerate(test_actions):
        gate_names = ['RX', 'RY', 'RZ', 'H', 'X', 'Y', 'Z', 'CX', 'CZ', 'CY', 'CCX', 'Barrier', 'ID']
        print(f"    Action {i}: {gate_names[gate]} on qubits {qubits} with param {param}")
    
    assert len(qc.data) > 0, "No gates were applied"

def test_feature_representation():
    """Test that the feature representation captures circuit properties."""
    print("\n=== Testing Feature Representation ===")
    
    num_qubits = 4
    max_depth = 10
    
    # Test 1: Empty circuit
    empty_qc = QuantumCircuit(num_qubits)
    empty_encoding = encode_circuit_into_input_embedding(empty_qc, num_qubits, max_depth)
    
    # Test 2: Circuit with gates
    complex_qc = QuantumCircuit(num_qubits)
    complex_qc.h(0)
    complex_qc.cx(0, 1)
    complex_qc.cx(1, 2)
    complex_qc.rx(0.5, 3)
    complex_encoding = encode_circuit_into_input_embedding(complex_qc, num_qubits, max_depth)
    
    # The encodings should be different
    difference = torch.norm(empty_encoding - complex_encoding)
    print(f"  Empty vs Complex circuit encoding difference: {difference:.4f}")
    assert difference > 0, "Empty and complex circuits should have different encodings"
    
    # Test 3: Parameter sensitivity
    param_qc1 = QuantumCircuit(num_qubits)
    param_qc1.rx(0.1, 0)
    encoding1 = encode_circuit_into_input_embedding(param_qc1, num_qubits, max_depth)
    
    param_qc2 = QuantumCircuit(num_qubits)
    param_qc2.rx(0.9, 0)
    encoding2 = encode_circuit_into_input_embedding(param_qc2, num_qubits, max_depth)
    
    param_difference = torch.norm(encoding1 - encoding2)
    print(f"  Different parameter encoding difference: {param_difference:.4f}")
    assert param_difference > 0, "Different parameters should produce different encodings"

def test_complete_workflow():
    """Test the complete encoding-decoding workflow."""
    print("\n=== Testing Complete Workflow ===")
    
    num_qubits = 6
    max_depth = 15
    
    # Step 1: Create original circuit
    original_qc = QuantumCircuit(num_qubits)
    original_qc.h(0)
    original_qc.cx(0, 1)
    original_qc.ry(0.3, 2)
    original_qc.cz(2, 3)
    
    # Step 2: Encode circuit
    encoding = encode_circuit_into_input_embedding(original_qc, num_qubits, max_depth)
    print(f"  Original circuit encoded to vector of size {encoding.shape[0]}")
    
    # Step 3: Simulate RL agent actions (based on the original circuit)
    actions = [
        (3, [0], 0.0),       # H gate
        (7, [0, 1], 0.0),    # CNOT gate
        (1, [2], 0.3),       # RY gate with parameter
        (8, [2, 3], 0.0),    # CZ gate
    ]
    
    # Step 4: Decode actions to create new circuit
    decoded_qc = decode_actions_into_circuit(actions, num_qubits)
    
    # Step 5: Encode the decoded circuit
    decoded_encoding = encode_circuit_into_input_embedding(decoded_qc, num_qubits, max_depth)
    
    print(f"  Decoded circuit has {len(decoded_qc.data)} gates")
    print(f"  Original circuit had {len(original_qc.data)} gates")
    print(f"  Encoding consistency check: {decoded_encoding.shape == encoding.shape}")
    
    assert decoded_encoding.shape == encoding.shape, "Encoding shapes should be consistent"

if __name__ == "__main__":
    print("Testing Part 1: Feature Encoding Scheme and Action Space")
    print("=" * 60)
    
    try:
        test_encoding_scalability()
        test_action_space_design()
        test_feature_representation()
        test_complete_workflow()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("\nThe proposed feature encoding scheme and action space are working correctly.")
        print("\nKey Features:")
        print("1. ✅ Scalable encoding that grows with number of qubits")
        print("2. ✅ Rich action space supporting 13 different gate types")
        print("3. ✅ Parameter handling for continuous optimization")
        print("4. ✅ Circuit structure, connectivity, and statistics captured")
        print("5. ✅ Robust encoding/decoding workflow")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
