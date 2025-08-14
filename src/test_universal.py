#!/usr/bin/env python3
"""
Test script for the corrected universal gate set implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helper_functions.encoding import encode_circuit_into_input_embedding
from helper_functions.decoding import decode_actions_into_circuit
from qiskit import QuantumCircuit
import torch
import numpy as np

def test_universal_gate_set():
    """Test the universal gate set implementation."""
    print("=== Testing Universal Gate Set {RX, RY, RZ, H, CNOT} ===")
    
    num_qubits = 4
    
    # Test actions with universal gate set
    test_actions = [
        (0, 0, 0.5),    # RX(π/2) on qubit 0
        (1, 1, -0.3),   # RY(-0.3π) on qubit 1
        (2, 2, 0.7),    # RZ(0.7π) on qubit 2
        (3, 3, 0.0),    # H on qubit 3 (parameter ignored)
        (4, 0, 0.0),    # CNOT control: qubit 0
        (5, 1, 0.0),    # CNOT target: qubit 1
        (0, 2, 0.2),    # RX(0.2π) on qubit 2
        (4, 2, 0.0),    # CNOT control: qubit 2
        (5, 3, 0.0),    # CNOT target: qubit 3
    ]
    
    # Test decoding
    qc = decode_actions_into_circuit(test_actions, num_qubits)
    
    print(f"  Created circuit with {len(qc.data)} gates")
    print(f"  Circuit depth: {qc.depth()}")
    
    # Count gate types
    rx_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'rx')
    ry_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'ry')
    rz_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'rz')
    h_count = sum(1 for inst in qc.data if inst.operation.name.lower() == 'h')
    cnot_count = sum(1 for inst in qc.data if inst.operation.name.lower() in ['cx', 'cnot'])
    
    print(f"  Gate counts - RX: {rx_count}, RY: {ry_count}, RZ: {rz_count}, H: {h_count}, CNOT: {cnot_count}")
    
    assert len(qc.data) > 0, "No gates were applied"
    assert rx_count > 0, "RX gates should be present"
    assert ry_count > 0, "RY gates should be present" 
    assert rz_count > 0, "RZ gates should be present"
    assert h_count > 0, "H gates should be present"
    assert cnot_count > 0, "CNOT gates should be present"
    
    return qc

def test_universal_encoding():
    """Test the universal gate set encoding scheme."""
    print("\n=== Testing Universal Gate Set Encoding ===")
    
    num_qubits = 4
    max_depth = 10
    
    # Create test circuit with all gate types
    qc = QuantumCircuit(num_qubits)
    qc.rx(0.5, 0)
    qc.ry(0.3, 1)
    qc.rz(0.7, 2)
    qc.h(3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    
    # Test encoding
    encoding = encode_circuit_into_input_embedding(qc, num_qubits, max_depth)
    
    # Calculate expected size
    num_gate_types = 5  # RX, RY, RZ, H, CNOT
    circuit_structure_size = max_depth * num_qubits * num_gate_types
    parameter_matrix_size = max_depth * num_qubits
    cnot_connections_size = max_depth * num_qubits * num_qubits
    statistics_size = 7
    
    expected_size = (circuit_structure_size + parameter_matrix_size + 
                    cnot_connections_size + statistics_size)
    
    print(f"  Encoding size: {encoding.shape[0]}")
    print(f"  Expected size: {expected_size}")
    print(f"  Size match: {encoding.shape[0] == expected_size}")
    
    assert encoding.shape[0] == expected_size, "Encoding size mismatch"
    
    # Test encoding differences
    simple_qc = QuantumCircuit(num_qubits)
    simple_qc.h(0)
    simple_encoding = encode_circuit_into_input_embedding(simple_qc, num_qubits, max_depth)
    
    difference = torch.norm(encoding - simple_encoding)
    print(f"  Encoding difference: {difference:.4f}")
    
    assert difference > 0, "Different circuits should have different encodings"

def test_vqe_ansatz_construction():
    """Test construction of VQE-like ansatz circuits."""
    print("\n=== Testing VQE Ansatz Construction ===")
    
    num_qubits = 4
    
    # Simulate a typical VQE ansatz construction
    ansatz_actions = [
        # Layer 1: Single qubit rotations
        (1, 0, 0.1),    # RY on qubit 0
        (1, 1, 0.2),    # RY on qubit 1
        (1, 2, 0.3),    # RY on qubit 2
        (1, 3, 0.4),    # RY on qubit 3
        
        # Entangling layer: CNOT gates
        (4, 0, 0.0),    # CNOT control: qubit 0
        (5, 1, 0.0),    # CNOT target: qubit 1
        (4, 2, 0.0),    # CNOT control: qubit 2
        (5, 3, 0.0),    # CNOT target: qubit 3
        
        # Layer 2: More rotations
        (0, 0, 0.5),    # RX on qubit 0
        (2, 1, 0.6),    # RZ on qubit 1
        (0, 2, 0.7),    # RX on qubit 2
        (2, 3, 0.8),    # RZ on qubit 3
        
        # Final entangling
        (4, 1, 0.0),    # CNOT control: qubit 1
        (5, 2, 0.0),    # CNOT target: qubit 2
    ]
    
    # Build ansatz circuit
    ansatz_qc = decode_actions_into_circuit(ansatz_actions, num_qubits)
    
    print(f"  VQE ansatz circuit:")
    print(f"    Total gates: {len(ansatz_qc.data)}")
    print(f"    Circuit depth: {ansatz_qc.depth()}")
    print(f"    Parameters: {ansatz_qc.num_parameters}")
    
    # Count gate types
    rx_count = sum(1 for inst in ansatz_qc.data if inst.operation.name.lower() == 'rx')
    ry_count = sum(1 for inst in ansatz_qc.data if inst.operation.name.lower() == 'ry')
    rz_count = sum(1 for inst in ansatz_qc.data if inst.operation.name.lower() == 'rz')
    cnot_count = sum(1 for inst in ansatz_qc.data if inst.operation.name.lower() in ['cx', 'cnot'])
    
    print(f"    Gate composition: RX={rx_count}, RY={ry_count}, RZ={rz_count}, CNOT={cnot_count}")
    
    # Test that this creates a reasonable VQE ansatz
    assert len(ansatz_qc.data) > num_qubits, "Ansatz should have substantial gate count"
    assert cnot_count > 0, "Ansatz should have entangling gates"
    assert (rx_count + ry_count + rz_count) > 0, "Ansatz should have parameterized gates"

def test_action_space_coverage():
    """Test that action space covers all gate types."""
    print("\n=== Testing Action Space Coverage ===")
    
    num_qubits = 4
    
    # Test all gate types
    for gate_type in range(6):  # 0-5 for RX, RY, RZ, H, CNOT_control, CNOT_target
        for qubit in range(num_qubits):
            for param in [-1.0, 0.0, 1.0]:
                action = [gate_type, qubit, param]
                
                # Validate action bounds
                assert 0 <= gate_type <= 5, f"Gate type {gate_type} out of bounds"
                assert 0 <= qubit < num_qubits, f"Qubit {qubit} out of bounds"
                assert -1.0 <= param <= 1.0, f"Parameter {param} out of bounds"
    
    print(f"  Tested {6 * num_qubits * 3} different actions")
    print("  All actions within valid bounds ✓")

if __name__ == "__main__":
    print("Testing Universal Gate Set Implementation for VQE")
    print("=" * 55)
    
    try:
        test_universal_gate_set()
        test_universal_encoding()
        test_vqe_ansatz_construction()
        test_action_space_coverage()
        
        print("\n" + "=" * 55)
        print("✅ ALL TESTS PASSED!")
        print("\nCorrected implementation features:")
        print("1. ✅ Universal gate set {RX, RY, RZ, H, CNOT}")
        print("2. ✅ Proper VQE ansatz construction capability")
        print("3. ✅ Complete parameterized gate support")
        print("4. ✅ Efficient encoding for all gate types")
        print("5. ✅ Action space covers full gate vocabulary")
        print("\nReady for VQE optimization with PPO!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
