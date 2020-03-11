# gene-expression-models
This repository provides functions implementing analytic and approximate methods developed in [1] to compute steady-state solutions of stochastic gene expression models. The repository consists of four key modules in the `source` directory: `analytic.py`, `extrinsic.py`, `FSP.py` and `recurrence.py`. Example scripts are also provided.

## Installation

A clean and complete installation can be achieved using conda and the provided `requirements.txt`, which contains the specification for a compatible set of packages. Alternatively, adding these same requirements to an existing python installation should also be straightforward. To install in a new conda environment named `gene-models`, open a terminal in the repository directory then implement the following steps.
```bash
conda create --prefix ./gene-models python=3.8
conda activate ./gene-models
conda install --file requirements.txt
```

The installation can then be tested by running one of the example scripts:
```bash
python examples/PlotRecurrence.py
```

## Examples
The folder `examples/` contains the following example scripts which can be run using the python interpreter:
- `PlotAnalytic.py`: computes and plots the analytic solution of several models.
- `PlotRecurrence.py`: approximate steady-state solutions using the recurrence method and compare finite finite-state projection algorithm.
- `PlotExtrinsic.py`: solutions for gene models with extrinsic noise on parameters using the analytic solution method and the recurrence method from [1].


## Common usage
Typical usage would be to generate the steady state probability distribution for a given gene expression model. In this case the distribution *p*(*n*) is evaluated for all copy numbers *n*=0:*N*-1, and returned as a list of length *N*. For example:
```python
import analytic as an
import matplotlib.pyplot as plt

prms = [0.1,0.1, 65.,15.]
N = 100
P = an.analytic_twostate(prms,N)
plt.plot(range(N), P)
```

Alternatively, The recurrence relation method enables the evaluation of the distribution at a single copy number *n*. This can be useful for example in parameter inference where a likelihood function is required for a finite number of samples. For example:
```python
import recurrence as rec
import math

samps = [2, 10, 4, 54, 12, 0, 3, 20]
prms = [0.1,0.1, 65.,15.]
M = 300 # Number of terms evaluated by the recursion relation.
G = recurrence_step_two_switch(prms, M)
L = 0.0
for s in samps: # Evaluate the log-likelihood
    L += math.log(rec.invgenfunc(G, s))
```

## Related literature
The analytic solutions and recurrence method implemented here are described in the paper:

[1] L. Ham, D. Schnoerr, R. D. Brackston & M. P. H. Stumpf, Exactly solvable models of stochastic gene expression. *bioRxiv* [2020.01.05.895359](https://doi.org/10.1101/2020.01.05.895359).
