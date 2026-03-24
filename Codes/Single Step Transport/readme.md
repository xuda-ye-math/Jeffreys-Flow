Here is the `README.md` file for your project.

---

# Jeffreys Flow Experiments on 2D Potentials

This repository contains the implementation for training and testing **Jeffreys Flow** (a Normalizing Flow trained via Jeffreys Divergence) on various 2D potential energy landscapes.

The codebase utilizes a hybrid workflow: **MATLAB** is used for generating ground truth data via Parallel Tempering and for final visualization, while **Python (PyTorch)** is used for training the Normalizing Flow models and generating samples.

## Prerequisites

* **MATLAB** (with Statistics and Machine Learning Toolbox)
* **Python 3.x**
* **PyTorch**
* **NumPy**, **SciPy**

## Usage Pipeline

Follow these steps in order to reproduce the experiments.

### 1. Configuration

Before running any script, ensure the target potential system is consistent across all files.

* **In Python:** Edit `parameters.py`. Set the `ABBR` variable to one of the supported system codes (e.g., `'TW'`, `'HB'`, `'AN'`, `'MW'`).
* **In MATLAB:** Edit `simulate_MU.m` and `plot_NU.m`. Ensure the `ABBR` variable matches the one set in Python.

**Supported Systems (`ABBR`):**

* `'TW'`: Three-Well Potential
* `'HB'`: Himmelblau Function
* `'AN'`: Annulus (Ring) Potential
* `'MW'`: Multiple Well Potential

### 2. Generate Training Data (MATLAB)

Run the script `simulate_MU.m`.

* This script performs Parallel Tempering (PT) MCMC sampling on the target potential.
* It generates the reference dataset () and saves it to the `./data` directory.

### 3. Train Flows (Python)

Run the script `train_flow.py`.

* This script loads the data generated in Step 2.
* It trains 5 separate Normalizing Flow models corresponding to different mixing parameters:

* Trained models are saved as `.pth` files in `./data`.

### 4. Generate Flow Samples (Python)

Run the script `generate_NU.py`.

* This script loads the 5 trained models from Step 3.
* It pushes samples from the base distribution through the flows to the target space.
* It computes importance weights and saves the results (`samples` and `weights`) to `.mat` files in `./data`.

### 5. Visualization and Analysis (MATLAB)

Run the script `plot_NU.m`.

* This script loads the reference data (from Step 2) and the generated flow data (from Step 4).
* It generates a 1x6 comparison plot showing:
* The Reference PT samples.
* Samples from all 5 flow strategies.


* It calculates and displays the **Effective Sample Size (ESS)** and **L2 Bias** for each method.
* The final high-resolution figure is saved to `./figures`.

## File Structure

* `parameters.py`: Global configuration file (System selection, hyperparameters).
* `model.py`: PyTorch implementation of the Normalizing Flow (Rational Quadratic Spline).
* `simulate_MU.m`: MATLAB script for ground truth data generation via Parallel Tempering.
* `train_flow.py`: Main training loop for different  values.
* `generate_NU.py`: Generates samples from trained models for evaluation.
* `plot_NU.m`: MATLAB script for plotting results and calculating metrics.
* `compute_bias.m`: Helper function to calculate the L2 Bias of observables.
* `potential/`: Directory containing class definitions for different potential energy landscapes (MATLAB `.m` and Python `.py`).