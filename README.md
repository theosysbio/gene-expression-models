# gene-expression-models
[![DOI](https://zenodo.org/badge/244686352.svg)](https://zenodo.org/badge/latestdoi/244686352)

This repository provides functions implementing analytic and approximate methods developed in [1] to compute steady-state solutions of stochastic gene expression models. The repository consists of four key modules in the `source` directory: `analytic.py`, `extrinsic.py`, `fsp.py` and `recurrence.py`. Example scripts are also provided.

## Installation

A clean and complete installation can be achieved using conda and the provided `requirements.txt`, which contains the specification for a compatible set of packages. Alternatively, adding these same requirements to an existing python installation should also be straightforward. To install in a new conda environment named `gene-models`, open a terminal in the repository directory then implement the following steps.
```bash
conda create --prefix ./gene-models python=3.8
conda activate ./gene-models
conda install --file requirements.txt
```

The installation can then be tested by running one of the pre-defined examples by running:
```bash
python main.py --method='analytic' --model='leaky'
```

## Examples
The folder `examples/` contains the following modules that implement functions that compute steady-state distributions of certain example systems for fixed parameters:
- `plot_analytic_solutions.py`: functions for computing and plotting the analytic solution of several models.
- `plot_recurrence_solutions.py`: functions for computing approximate steady-state solutions using the recurrence method and compare finite finite-state projection algorithm.
- `plot_extrinsic_solutions.py`: functions for computing solutions for gene models with extrinsic noise on parameters using the analytic solution method and the recurrence method from [1].

The following models are available for the different methods:

- analytic method ('analytic'): 
  * leaky Telegraph model ('leaky')
  * 2^2 model ('twotwo')
  * 2^3 model ('twothree')
- recurrence method ('recurrence'):
    * leaky Telegraph model ('leaky')
    * three switch model ('three_switch')
    * feedback model ('feedback')
- models with extrinsic noise ('extrinsic'):
    * leaky Telegraph model ('leaky')
    * three switch model ('three_switch')
    * 2^2 model ('twotwo')

The string in parentheses can be used to call one of the methods to solve and plot one of the available models from the command line (as an example the analytic method gets called here for the leaky Telegraph model):
```bash
python main.py --method='analytic' --model='leaky'
```

## Common usage
For a more refined analysis the modules in source/ can be used. Typical usage would be to generate the steady-state probability distribution for a given gene expression model. In this case the distribution *p*(*n*) is evaluated for all copy numbers *n*=0:*N*-1, and returned as a list of length *N*. For example, the following code computes and plots the analytic steady-state solution of the two-state (leaky Telegraph) model for *N=100*:
```python
import source.analytic as an
import matplotlib.pyplot as plt

prms = [0.1, 0.1, 65., 15.]
N = 100
P = an.analytic_twostate(prms,N)
plt.plot(range(N), P)
plt.show()
```

Alternatively, The recurrence relation method enables the evaluation of the distribution at a single copy number *n*. This can be useful for example in parameter inference where a likelihood function is required for a finite number of samples. For example:
```python
import source.recurrence as rec
import math

samps = [2, 10, 4, 54, 12, 0, 3, 20]
prms = [0.1,0.1, 65.,15.]
M = 300 # Number of terms evaluated by the recursion relation.
G = rec.recurrence_step_two_switch(prms, M)
L = 0.0
for s in samps: # Evaluate the log-likelihood
    L += math.log(rec.invgenfunc(G, s))
```

## Related literature
The analytic solutions and recurrence method implemented here are described in the paper:

[1] L. Ham, D. Schnoerr, R. D. Brackston & M. P. H. Stumpf, [Exactly solvable models of stochastic gene expression](https://doi.org/10.1063/1.5143540), J. Chem. Phys. 152, 144106 (2020).
