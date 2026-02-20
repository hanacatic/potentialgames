# Finite-time convergence to an ϵ-efficient Nash equilibrium in potential games
<!-- This repository is part of a Semester project with SYCAMORE (Systems Control and Multiagent Optimization Research) Laboratory at EPFL. It contains implementations of log-linear learning and its modifications as part of validation of alogrithm proposed in [[1]](#1). -->

This repository accompanies the paper:

Maddux, A., Ouhamma, R., Catic, H., and Kamgarpour, M., "Finite-time convergence to an ϵ-efficient Nash equilibrium in potential games", IEEE Transactions on Control of Network Systems (2026).

## Dependencies
The project was tested on Windows 11 and is written in Python 3.11. Dependencies are managed via Poetry (see `pyproject.toml`).

## Setup

1. Install Git LFS. Preprocessed experiment data are tracked using Git LFS.
    ```
    git lfs install
    ```
2. Clone the repository:
    ```
    git clone https://github.com/hanacatic/potentialgames.git
    cd potentialgames
    ```
3. Install dependencies:
    ```
    pipx install poetry
    poetry install
    ```

## Usage

Scripts and modules for running experiments are provided in the `scripts/` and functions to visualise results are provided in the `potentialgames/utils/` directories. See the code and docstrings for details.

### Quick example 
    ```
    poetry run python main.py
    ```

<!-- ## References
<a id="1">[1]</a> 
Maddux, A., Ouhamma, R., Catic, H., and Kamgarpour, M., (2024). 
Finite time convergence to ε-efficient Nash Equilibirum in potential games. 
arXiv preprint arXiv:2405.15497. -->
