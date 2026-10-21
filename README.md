<<<<<<< HEAD
# Qgrad

This repository contains the code used to generate the figures for the paper [*Parameter-Shift Rules for Gradients in Boson Sampling Experiments*](https://arxiv.org/abs/2607.15160), available on arXiv.

The repository implements the Parameter-Shift Rule (PSR) and the Finite Difference (FD) method for calculating gradients in photonic quantum circuits. It uses [Perceval](https://github.com/Quandela/Perceval), a software platform for discrete-variable photonic quantum computing [1], making the code compatible with Quandela's Ascella and Belenos quantum processing units (QPUs).

The interferometer circuits are constructed using the Clements decomposition [2], implemented in `clements.py`. The simulated implementations of the PSR and FD methods are provided in `qdiff_theo.py`, while `qpu.py` contains the code required to run experiments on a physical QPU.

The figures presented in the paper are generated using the following Jupyter notebooks:

- `exp_grad.ipynb`: Experimental results obtained using a photonic QPU.
- `sim_grad.ipynb`: Numerical simulations comparing the gradient estimation methods.
- `theo_grad.ipynb`: Theoretical results and analysis.

## Requirements

The requirements are listed in the requirements.txt file. 

Running experiments on a physical QPU additionally requires access to the corresponding Quandela Cloud platform and the creation of a valid token.

## References

[1] N. Heurtel et al., "Perceval: A Software Platform for Discrete Variable Photonic Quantum Computing," *Quantum*, vol. 7, p. 931, 2023. [https://doi.org/10.22331/q-2023-02-21-931](https://doi.org/10.22331/q-2023-02-21-931).

[2] W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, and I. A. Walmsley, "Optimal design for universal multiport interferometers," *Optica*, vol. 3, no. 12, pp. 1460–1465, 2016. [https://doi.org/10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460).
=======
# Qgrad
>>>>>>> b5374cab7574a4f1b8e23238c4b7d5c9ac114d87
