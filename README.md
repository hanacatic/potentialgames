# Finding ε-efficient Nash Equilibirum in potential games

This repository is part of a Semester project with SYCAMORE (Systems Control and Multiagent Optimization Research) Laboratory at EPFL. It contains implementations of log-linear learning and its modifications as part of validation of alogrithm proposed in [[1]](#1).

## Dependencies
The project was tested on Windows 11 and is written in Python 3.12. Other requirements include:
* numpy
* matplotlib
* pandas
* scipy
* networkx
* [SciencePlots](https://github.com/garrettj403/SciencePlots)

Recommended:
* pytest (for running tests)

## Setup

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/potentialgames.git
    cd potentialgames
    ```
2. Install dependencies:
    ```
    pip install -r [requirements.txt]
    ```

## Usage

Scripts and modules for running experiments and plotting results are provided in the `scripts/` and `potentialgames/` directories. See the code and docstrings for details.


## References
<a id="1">[1]</a> 
Maddux, A., Ouhamma, R. and Kamgarpour, M., (2024). 
Finite time convergence to ε-efficient Nash Equilibirum in potential games. 
arXiv preprint arXiv:2405.15497.
