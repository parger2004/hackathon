# Setup Instructions

## Environment Setup

A Python virtual environment has been created and configured with all necessary dependencies.

### Prerequisites
- Python 3.12+
- pip

### Environment Creation and Activation

1. **Activate the virtual environment:**
   ```bash
   cd /home/nick/Documents/Hackathon/qiskit-hackathon-taiwan-2025
   source venv/bin/activate
   ```

### Installed Packages

The following packages have been installed:
- **PyTorch**: For deep learning and neural networks
- **Qiskit 1.3.1**: For quantum computing
- **Qiskit Nature 0.7.2**: For quantum chemistry applications
- **Qiskit Algorithms 0.3.1**: For quantum algorithms
- **PySCF**: For quantum chemistry calculations
- **NumPy, SciPy, Matplotlib**: For scientific computing
- **Gymnasium**: For reinforcement learning environments
- **Jupyter**: For notebook support

### Import Issues Fixed

Fixed import statements in the following files to work with the current project structure:
- `src/main.py`: Changed `from src.` imports to relative imports
- `src/agent.py`: Fixed imports for actor-critic networks and memory
- `src/env.py`: Fixed imports for helper functions

### Running the Project

To run the main script:

```bash
cd /home/nick/Documents/Hackathon/qiskit-hackathon-taiwan-2025/src
source ../venv/bin/activate
python main.py config_lih.cfg
```

The script successfully:
- Loads the configuration from `config_lih.cfg`
- Loads the qubit operator from `./operators/qubit_op_LiH.qpy`
- Initializes the VQE environment with molecular parameters
- Creates the PPO agent
- Sets up the training loop (currently with a placeholder `break` statement)

### Configuration

The project uses configuration files (`.cfg`) to specify molecular and training parameters. The example configuration `config_lih.cfg` is set up for LiH (Lithium Hydride) molecule calculations.

### Next Steps

The environment is now ready for development. You can:
1. Implement the training loop in `main.py`
2. Complete the PPO agent methods in `agent.py`
3. Modify the environment behavior in `env.py`
4. Add new molecular configurations by creating new `.cfg` files
