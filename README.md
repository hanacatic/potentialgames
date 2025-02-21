# Finding ε-efficient Nash Equilibirum in potential games

This repository is part of a Semester project with SYCAMORE (Systems Control and Multiagent Optimization Research) Laboratory at EPFL. It contains implementations of log-linear learning and its modifications as part of validation of alogrithm proposed in [[1]](#1).

## Dependencies
The project was tested on Windows 11. There are no known issues with other Operating Systems. It is written in Python 
3.12, other requirements include:
* numpy
* matplotlib
* pandas
* scipy
* networkx
* [Transportation Networks](https://github.com/bstabler/TransportationNetworks)

Recommended:
* Cython (For this C/C++ compiler is required)



## Setup

If you want to levarage the computational efficiency of precomputed C code with Cython, in lib, lib/games and lib/aux_functions there are setup files which will compile the Python code into C, when following command is run in these folders.

```
python setup.py build_ext --inplace 
```
Running this command will generate .c, .pyd and build files for the desired .py files. Classes and packages can be used as usual after this. 

Note that if there are any changes to the code after running this command they will not be visible unless the code is recompiled. In case there is a need for changing these files often, it is suggested to skip this step for now, or exclude the files that will be changed by adjusting the setup.py file.

## References
<a id="1">[1]</a> 
Maddux, A., Ouhamma, R. and Kamgarpour, M., (2024). 
Finite time convergence to ε-efficient Nash Equilibirum in potential games. 
arXiv preprint arXiv:2405.15497.