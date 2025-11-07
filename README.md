# GEORCE-FM: Simultaneous Optimization of Geodesics and Fréchet Means
![conceptual_riemannian_frechet_mod2.pdf](https://github.com/user-attachments/files/23414078/conceptual_riemannian_frechet_mod2.pdf)

This repository shows how to jointly estimate geodesics and the Fréchet mean to reduce the computational complexity of computing the Fréchet mean on non-trivial Riemannian and Finslerian manifolds.

## Installation and Requirements

The implementations in the GitHub is Python using JAX. To clone the GitHub reporsitory and install packages type the following in the terminal

```
git clone https://github.com/FrederikMR/georce.git
cd georce
pip install -r requirements.txt
```

The first line clones the repository, the second line moves you to the location of the files, while the last line install the packages used in repository.

## Code Structure

The following shows the structure of the code. All general implementations of geometry and optimization algorithms can be found in the "geometry" folder for both the Riemannian and Finsler case.

    .
    ├── data                               # Contains generated data for experiments
    ├── geometry                           # Contains implementation of Finsler and Riemannian manifolds as well as geodesic optimization algorithms, inlcuding GEORCE_FM
    ├── models                             # Contains the parameters of the trained VAEs
    ├── timing_gpu                         # Contains all timing results on a GPU
    ├── timing_cpu                         # Contains all timing results on a CPU
    ├── vae                                # Contains the architecture of the used VAEs
    ├── finsler_frechet.ipynb              # Runs all figures for the Finsler case
    ├── generate_data.py                   # Generates all synthetic data
    ├── load_manifold.py                   # Automatically loads a manifold structure
    ├── riemannian_frechet.ipynb           # Runs all figures for the Riemannian case
    ├── runtime.py                         # Runs all runtime computations
    ├── runtime_adaptive.py                # Runtime for adaptive algorithms
    ├── runtime_estimates.ipynb            # Plots and display of computed runtimes
    ├── train_vae.py                       # Script for training the VAE's
    ├── vae_frechet.py                     # Plotting of the figures for the VAEs
    └── README.md

The remaining files and folders are used for submitting code to a GPU and various tests.

## Reproducing Experiments

All experiments can be re-produced by running the notebooks and the runtime.py package for the given manifold, hyper-parameters and optimization method.

## Logging

All experimental results for the runtime and length estimates are saved as .pkl files in the folder "timing".

## Reference

If you want to use GEORCE_FM for scientific purposes, please cite:

    @misc{rygaard2025simultaneousoptimizationgeodesicsfrechet,
      title={Simultaneous Optimization of Geodesics and Fr\'echet Means}, 
      author={Frederik Möbius Rygaard and Søren Hauberg and Steen Markvorsen},
      year={2025},
      eprint={2511.04301},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2511.04301}, 
      }


