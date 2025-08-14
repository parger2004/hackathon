from typing import List, Tuple, Union, Optional  # For typing annotation.
import gymnasium as gym                          # For open-ai gym compatibility.
import numpy as np
import warnings                                  # To ignore warnings.
import torch                                     # For tensor manipulation (state and action space).
import time                                      # For time tracking.
import os                                        # For file system access.

# Custom helper functions for the environment:
from helper_functions.decoding import decode_actions_into_circuit
from helper_functions.encoding import encode_circuit_into_input_embedding  

# Quantum Circuits:
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter # To build quantum circuits.
from qiskit import qpy                                                # To save quantum circuits to Disk.

# Quantum Chemistry:
from qiskit_nature.second_q.drivers import PySCFDriver                             # Driver for obtaining molecular information using PySCF.
from qiskit_nature.second_q.circuit.library import HartreeFock                     # Constructs the Hartree-Fock initial state.
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer              # Freezes core spatial orbitals to reduce the molecule size in quantum simulations.
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo              # Data structure for storing molecule information.
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver         # Classical solver for finding the exact minimum eigenvalue of a Hamiltonian.
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp # Used for type annotation.
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper # Maps fermionic operators to spin qubit-equivalent operators.

#--------------------------------
# Estimators for computing the expectation value of the Hamiltonian.

# For exact theoretical calculations using the statevector simulation without noise or statistical sampling:
from qiskit.primitives import StatevectorEstimator # Docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator

# For high-performance simulations with Aer when testing circuits with noise models.
#from qiskit_aer.primitives import Estimator as aer_estimator # Docs: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.Estimator.html
#--------------------------------

# Ignore warnings:
warnings.filterwarnings("ignore")

class VQEnv(gym.Env):
    '''
    - Args:
        - molecule_name (str): the name of the molecule.
        - mapper_name (str): the name of the mapper to use for converting fermionic operators to qubit operators.
        - conv_tol (float): the convergence tolerance.
        - max_circuit_depth (int): the maximum number of circuit layers.
        - max_steps_per_episode (int): the maximum number of iterations per episode.
        - symbols (list): the molecule's symbols.
        - geometry (tuple): the molecule's coordinates.
        - multiplicity (int): the molecule's multiplicity: 2*spin + 1.
        - charge (int): the molecule's charge.
        - num_electrons (int): the number of electrons to use in the calculation.
        - num_spatial_orbitals (int): the molecule's number of spatial orbitals can be provided or found.
        - num_particles (tuple): the molecule's number of particles can be provided or found.
        - fci_energy (float): the molecule's ground state energy can be provided or found.
        - qubit_operator (SparsePauliOp): the molecule's qubit-equivalent Hamiltonian can be provided or found.
        - num_qubits (int): the molecule's number of qubits can be provided or found.
    '''
    def __init__(self,
                 # Hyperparameters:
                 molecule_name: str = "Unknown",
                 mapper_name: str = "Jordan-Wigner",
                 basis: str = "sto3g",
                 conv_tol: float = 1e-5,
                 max_circuit_depth: int = 50,
                 max_steps_per_episode: int = 20,

                 # Molecule information:
                 symbols: Optional[List[str]] = None,
                 geometry: Optional[Tuple[List[float], ... ]] = None,
                 multiplicity: Optional[int] = 1,
                 charge: Optional[int] = 0,
                 num_electrons: Optional[int] = None,  
                 num_spatial_orbitals: Optional[int] = None,
                 num_particles: Optional[Tuple[int, ... ]] = None,
                 fci_energy: Optional[float] = None,

                 # Qubit operator:
                 qubit_operator: Optional[SparsePauliOp] = None,
                 num_qubits: Optional[int] = None,
                 ):

        # Run the constructor of the parent class (gym.Env):
        super().__init__()

        # Hyperparameters:
        self.molecule_name = molecule_name
        self.conv_tol = conv_tol
        self.max_circuit_depth = max_circuit_depth
        self.max_steps_per_episode = max_steps_per_episode

        # Mapper:
        match mapper_name:
            case "Jordan-Wigner":
                mapper = JordanWignerMapper()
            case "Bravyi-Kitaev":
                mapper = BravyiKitaevMapper()
            case "Parity":
                mapper = ParityMapper(num_particles=num_particles)
            case _:
                raise ValueError(f"Unknown mapper_name: {mapper_name}")

        # If the qubit operator was not provided:
        if qubit_operator is None:
            # Input Validation:
            if not isinstance(symbols, list):
                raise ValueError("Something is missing. The symbols argument must be provided as a list of strings.")
            if not isinstance(geometry, tuple):
                raise ValueError("Something is missing. The geometry argument must be provided as a tuple of lists of floats.")
            if not isinstance(multiplicity, int):
                raise ValueError("Something is missing. The multiplicity argument must be provided as a type int.")
            if not isinstance(charge, int):
                raise ValueError("Something is missing. The charge argument must be provided as a type int.")
            # Get molecule properties:
            (
                self.qubit_operator, 
                num_particles, 
                num_spatial_orbitals, 
                molecule,
            ) = self.get_qubit_op(
                symbols = symbols,
                geometry = geometry,
                multiplicity = multiplicity, 
                charge = charge,
                basis = basis,
                num_electrons_to_use = num_electrons,
                num_spatial_orbitals_to_use = num_spatial_orbitals,
                mapper = mapper
            )
            self.fci_energy = fci_energy or self.get_fci_energy(self.qubit_operator, molecule = molecule)
            self.num_qubits = num_qubits or self.qubit_operator.num_qubits
        # If the qubit operator was provided:
        else:
            # Input Validation:
            if not isinstance(qubit_operator, SparsePauliOp):
                raise ValueError("The qubit_operator argument must be provided as a Qiskit SparsePauliOp object.")
            if not isinstance(num_spatial_orbitals, int):
                raise ValueError("The num_spatial_orbitals argument must be provided as a type int.")
            if not isinstance(fci_energy, float):
                raise ValueError("The fci_energy argument must be provided as a type float.")
            self.qubit_operator = qubit_operator
            self.fci_energy = fci_energy
            self.num_qubits = num_qubits or self.qubit_operator.num_qubits

        # Reward range:
        self.reward_range = (-float('inf'), float('inf'))  # [min, max]

        # Reference Hartree-Fock state:
        self.HF = HartreeFock(num_spatial_orbitals, num_particles, mapper)

        # Universal Action space for VQE: {RX, RY, RZ, H, CNOT}
        # Action vector: [gate_type, qubit_index, parameter_value]
        # - gate_type: 0=RX, 1=RY, 2=RZ, 3=H, 4=CNOT_control, 5=CNOT_target
        # - qubit_index: target qubit index [0, num_qubits-1]
        # - parameter_value: rotation angle for parameterized gates [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, -1.0]), 
            high=np.array([5, self.num_qubits-1, 1.0]), 
            dtype=np.float32
        )

        # Observation space: Based on universal gate set encoding
        # Calculate the size of the encoded circuit representation
        num_gate_types = 5  # RX, RY, RZ, H, CNOT
        circuit_structure_size = max_circuit_depth * self.num_qubits * num_gate_types
        parameter_matrix_size = max_circuit_depth * self.num_qubits
        cnot_connections_size = max_circuit_depth * self.num_qubits * self.num_qubits
        statistics_size = 7
        
        total_state_size = (circuit_structure_size + 
                           parameter_matrix_size + 
                           cnot_connections_size + 
                           statistics_size)
        
        self.observation_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(total_state_size,), 
            dtype=np.float32
        )

        # Counter for rendering:
        self.render_counter = 0

        # Metadata:
        self.metadata = {
            'render_modes': ['circuit', 'energy', 'reward'],
            'render_fps': 60,
        }

        print(f"\nNumber of spatial orbitals: {num_spatial_orbitals}")
        print(f"Number of particles: {num_particles}")
        print(f"Number of qubits: {self.num_qubits}")
        print(f'FCI Energy: {self.fci_energy}')

    def get_qubit_op(self, 
                     symbols, 
                     geometry, 
                     multiplicity, 
                     charge, 
                     basis,
                     num_electrons_to_use, 
                     num_spatial_orbitals_to_use,
                     mapper):
        """
        Get the qubit-equivalent Hamiltonian of a particular molecule.

        Args:
            symbols (list): The chemical symbols of the atoms in the molecule.
            geometry (tuple): The coordinates of the atoms in the molecule.
            multiplicity (int): The multiplicity of the molecule (2*spin + 1).
            charge (int): The charge of the molecule.
            basis (str): The basis set to use for the molecular calculation.
            num_electrons_to_use (int): The number of electrons to use in the calculation.
            num_spatial_orbitals_to_use (int): The number of spatial orbitals to use in the calculation.
            mapper: The mapper to use for converting fermionic operators to qubit operators.

        Returns:
            qubit_op (SparsePauliOp): The qubit operator representing the Hamiltonian.
            num_particles (tuple): The number of particles in the molecule.
            num_spatial_orbitals (int): The number of spatial orbitals in the molecule.
            molecule (MoleculeInfo): The molecule information after freezing orbitals.
        """
        print(f'\nBuilding the qubit-equivalent Hamiltonian for the {self.molecule_name} molecule...')

        z2symmetry_reduction = None

        # Mol. info with coordinates in Angstrom:
        molecule_info = MoleculeInfo(
            symbols=symbols,
            coords=geometry,
            multiplicity=multiplicity,
            charge=charge,
        )

        # Driver:
        driver = PySCFDriver.from_molecule(molecule_info, basis=basis)

        # Get the electronic structure of the molecule:
        molecule = driver.run()

        #print('Num of alpha+beta spin electrons before reduction:', molecule.num_alpha + molecule.num_beta)
        #print('Num of spatial orbitals before reduction:', molecule.num_spatial_orbitals)

        # Reduce the molecule size using ACTIVE space approximation:
        transformer = ActiveSpaceTransformer(
            num_electrons=num_electrons_to_use,
            num_spatial_orbitals=num_spatial_orbitals_to_use,
            active_orbitals=None
        )

        # Reduced electronic structure of the molecule:
        molecule  = transformer.transform(molecule)

        #print('\nNum of alpha+beta spin electrons after reduction:', molecule.num_alpha + molecule.num_beta)
        #print('Num of spatial orbitals after reduction:', molecule.num_spatial_orbitals)

        # Properties:
        num_particles = molecule.num_particles
        num_spatial_orbitals = molecule.num_spatial_orbitals

        # Get the Operator in the 2nd quantization formalism:
        second_q_ops = molecule.second_q_ops()

        # Get the Hamiltonian:
        hamiltonian = second_q_ops[0]

        # Get the qubit operator:
        qubit_op = mapper.map(hamiltonian)

        # Apply symmetry:
        if z2symmetry_reduction != None:
            tapered_mapper = molecule.get_tapered_mapper(mapper)
            qubit_op = tapered_mapper.map(hamiltonian)

        return qubit_op, num_particles, num_spatial_orbitals, molecule

    def get_fci_energy(self, qubit_op, molecule):
        """
        Get the ground state energy of a particular molecule.

        Args:
            qubit_op (SparsePauliOp): The qubit operator representing the Hamiltonian.
            molecule (MoleculeInfo): The molecule.

        Returns:
            float: The ground state energy of the molecule.
        """

        sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
        result = molecule.interpret(sol)
        return result.total_energies[0].real

    def compute_expectation_value(self, ansatz, hamiltonian, params) -> float:
        """
        Compute the expectation value of the Hamiltonian given a custom ansatz and parameters.

        Args:
            ansatz (QuantumCircuit): The quantum circuit representing the ansatz.
            hamiltonian (SparsePauliOp): The Hamiltonian as a SparsePauliOp object.
            params (np.ndarray or torch.Tensor): Parameter values for the ansatz.

        Returns:
            float: The computed expectation value.
        """

        # Estimator:
        estimator = StatevectorEstimator()
        # Pubs:
        pub = (ansatz, hamiltonian, params)
        # Run:
        job = estimator.run([pub])
        result = job.result()[0]
        expectation_value = result.data.evs

        return expectation_value

    def get_expectation_value(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
        """
        Get the expectation value (energy) of a quantum circuit with respect to a Hamiltonian.
        
        Args:
            circuit (QuantumCircuit): The quantum circuit (ansatz).
            hamiltonian (SparsePauliOp): The molecular Hamiltonian.
            
        Returns:
            float: The expectation value (energy).
        """
        try:
            # Use the existing compute_expectation_value method
            return self.compute_expectation_value(circuit, hamiltonian, [])
        except Exception as e:
            # Return a large positive energy for invalid circuits
            return 1000.0

    def compute_reward(self, qc: QuantumCircuit = None):
        """
        Computes the reward for a given circuit based on VQE energy.
        FIXED: Multi-level reward shaping for better learning signal.

        Args:
            qc (QuantumCircuit): The quantum circuit to evaluate. If None, uses self.current_circuit.

        Returns:
            reward (float): the reward value based on energy improvement.
        """
        
        if qc is None:
            qc = self.current_circuit
            
        try:
            # Compute expectation value using the circuit
            energy = self.get_expectation_value(qc, self.qubit_operator)
            
            # Calculate energy difference from target FCI energy
            energy_diff = abs(energy - self.fci_energy)
            
            # FIXED: Multi-tier reward system with positive reinforcement
            if energy_diff < self.conv_tol:  # Converged (< 1e-5)
                reward = 1000.0
                print(f"  🎉 CONVERGENCE ACHIEVED! Energy diff: {energy_diff:.2e}")
            elif energy_diff < 0.001:  # Very close (< 1e-3)
                reward = 500.0
                print(f"  🌟 Very close to target! Energy diff: {energy_diff:.4f}")
            elif energy_diff < 0.01:   # Close (< 1e-2)
                reward = 100.0
                print(f"  ⭐ Getting close! Energy diff: {energy_diff:.4f}")
            elif energy_diff < 0.1:    # Progress (< 1e-1)
                reward = 20.0
                print(f"  📈 Making progress! Energy diff: {energy_diff:.4f}")
            elif energy_diff < 0.5:    # Some improvement
                reward = 5.0
            else:
                # Shaped reward for large errors - still provide learning signal
                max_reasonable_diff = 2.0  # Reasonable max for LiH molecule
                progress_ratio = max(0, (max_reasonable_diff - energy_diff) / max_reasonable_diff)
                reward = progress_ratio * 10 - 5  # Range: [-5, +10]
            
            # Efficiency bonus: encourage shorter circuits
            circuit_depth = qc.depth()
            if circuit_depth <= 3:
                efficiency_bonus = 5.0  # Bonus for very efficient circuits
            elif circuit_depth <= 10:
                efficiency_bonus = 2.0  # Small bonus for reasonable circuits
            else:
                efficiency_bonus = -0.1 * (circuit_depth - 10)  # Penalty for very deep circuits
            
            # Gate count bonus: encourage fewer gates
            gate_count = len(qc.data)
            if gate_count <= 10:
                gate_bonus = 2.0
            elif gate_count <= 20:
                gate_bonus = 1.0
            else:
                gate_bonus = -0.05 * (gate_count - 20)
            
            total_reward = reward + efficiency_bonus + gate_bonus
            
            # Store energy for tracking
            if not hasattr(self, 'energy_history'):
                self.energy_history = []
            self.energy_history.append(energy)
            
            return total_reward
            
        except Exception as e:
            print(f"  ❌ Circuit evaluation failed: {e}")
            return -100.0  # Reduced penalty for errors

    def reset(self, seed: int = 42, options: dict = {}) -> tuple:
        """
        Reset the environment to the initial state.

        - Args:
            - seed (int): random seed.
            - options (dict): dictionary of options with additional information of how to reset the environment.
        
        - Returns:
            - state (numpy.ndarray): the initial state of the environment.
            - info (dict): additional information if required.
        """
        
        print('\nReseting the environment...')
        # Seed for reproducibility, i.e., to generate the same initial state for each episode:
        np.random.seed(seed)
        # Terminal state reached:
        self.terminated = False
        # Max episode length (time steps) reached:
        self.truncated = False
        # Counter for the Maximum number of steps in the episode:
        self.counter = 0
        
        # Initialize the quantum circuit with Hartree-Fock state
        self.current_circuit = self.HF.copy()
        
        # Initialize the environment state using circuit encoding
        self.state = encode_circuit_into_input_embedding(
            self.current_circuit, 
            self.num_qubits, 
            self.max_circuit_depth
        ).numpy().astype(np.float32)
        
        self.info = {'ep_reward': [], 'ep_energy': []}
        return self.state, self.info
    
    def step(self, action: np.ndarray) -> tuple:
        """
        Returns a single experience from the environment.

        - Args:
            - action (np.ndarray): action taken by the agent [gate_type, qubit_index, parameter].

        - Returns:
            - self.new_state (numpy.ndarray): the next state.
            - self.reward (float): reward for the action taken.
            - self.terminated (bool): whether the episode is terminated.
            - self.truncated (bool): whether the episode is truncated.
            - self.info (dict): to ensure gym-compliance.
        """

        # Update counter
        self.counter += 1

        # Parse action: [gate_type, qubit_index, parameter_value]
        gate_type = int(np.round(np.clip(action[0], 0, 5)))
        qubit_index = int(np.round(np.clip(action[1], 0, self.num_qubits - 1)))
        parameter_value = np.clip(action[2], -1.0, 1.0)
        
        # Convert action to the format expected by decode_actions_into_circuit
        action_tuple = (gate_type, qubit_index, parameter_value)
        
        # Apply the action to build the circuit
        try:
            # Get current circuit and add new gate
            new_circuit = decode_actions_into_circuit([action_tuple], self.num_qubits, self.current_circuit)
            
            # Compute reward based on energy
            self.reward = self.compute_reward(new_circuit)
            
            # Update current circuit
            self.current_circuit = new_circuit
            
            # Encode the new circuit into state representation
            self.new_state = encode_circuit_into_input_embedding(
                self.current_circuit, 
                self.num_qubits, 
                self.max_circuit_depth
            ).numpy().astype(np.float32)
            
            # Get current energy for tracking
            current_energy = self.get_expectation_value(self.current_circuit, self.qubit_operator)
            
            # Check termination conditions
            energy_diff = abs(current_energy - self.fci_energy)
            
            # Terminated if converged within tolerance
            self.terminated = (energy_diff < self.conv_tol)
            
            # Truncated if max steps reached or circuit too deep
            self.truncated = (self.counter >= self.max_steps_per_episode or 
                            self.current_circuit.depth() >= self.max_circuit_depth)
            
            # Update info
            self.info['ep_reward'].append(self.reward)
            self.info['ep_energy'].append(current_energy)
            
            # Update state for next iteration
            self.state = self.new_state
            
        except Exception as e:
            # Handle invalid actions
            self.reward = -1000.0
            self.new_state = self.state.copy()  # Keep current state
            self.terminated = False
            self.truncated = False
            
            self.info['ep_reward'].append(self.reward)
            self.info['ep_energy'].append(float('inf'))

        return self.new_state, self.reward, self.terminated, self.truncated, self.info

    def render(self, mode: list = ['circuit', 'energy', 'reward'], flag: str = 'inference'):
        '''
        Plot the optimized circuit, reward curve and energy curve in a .png file.
        '''
        if self.state is None:
            raise ValueError('Failed to render. The environment is not initialized yet. Call the reset() method first.')

        def _render_circuit():
            save_dir = f'./results/{self.mol_name}/{flag}/imgs/optimized_circuit/'
            os.makedirs(save_dir, exist_ok=True)

        def _render_energy():
            save_dir = f'./results/{self.mol_name}/{flag}/imgs/energy_curve/'
            os.makedirs(save_dir, exist_ok=True)
            self.info['ep_energy'].append(self.current_energy)

        def _render_reward():
            save_dir = f'./results/{self.mol_name}/{flag}/imgs/reward_trend/'
            os.makedirs(save_dir, exist_ok=True)
            self.info['ep_reward'].append(self.reward)

        for m in mode:
            print(f'Rendering {m}...')
            if m not in self.metadata['render_modes']:
                raise ValueError(f'Invalid mode. Choose one of the following: {self.metadata["render_modes"]}')
            elif m == 'circuit':
                _render_circuit()
            elif m == 'energy':
                _render_energy()
            elif m == 'reward':
                _render_reward()

        self.render_counter += 1
    
    def close(self):
        """
        (Optional) Close the environment.
        """

        '''
        Write your code here.
        '''

        pass