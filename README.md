# Reliability and Latency Analysis for Wireless Communication Systems with a Secret-Key Budget

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/secret-key-budget-ruin/HEAD)
![GitHub](https://img.shields.io/github/license/klb2/secret-key-budget-ruin)
[![arXiv](https://img.shields.io/badge/arXiv-2304.02538-informational)](https://arxiv.org/abs/2304.02538)


This repository is accompanying the paper "Reliability and Latency Analysis for
Wireless Communication Systems with a Secret-Key Budget" (Karl-L. Besser,
Rafael Schaefer, and Vincent Poor, Apr. 2023.
[arXiv:2304.02538](https://arxiv.org/abs/2304.02538)).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.


## File List
The following files are provided in this repository:

- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `util.py`: Python module that contains utility functions, e.g., for saving results.
- `illustration_budget.py`: Exemplary illustration of the temporal progress of
  the considered secret-key budget
- `simulation_deterministic.py`: Simulation of the deterministic timing scheme
  (alternating between SKG and TX)
- `simulation_random_timing.py`: Simulation of the random timing scheme
- `ide_stopping_time.py`: Python module that contains functions to calculate
  the stopping times numerically via solving an IDE.
- `mc_stopping_time.py`: Python module that contains functions to determine the
  stopping times via Monte Carlo simulations.
- `bounds_stopping_time.py`: Python module that contains functions to calculate
  bounds on the stopping times.
- `ultimate_ruin_prob.py`: Python module that contains functions to calculate
  the probability of ultimate ruin (for the random timing scheme).


## Usage
### Running it online
You can use services like [CodeOcean](https://codeocean.com) or
[Binder](https://mybinder.org/v2/gh/klb2/secret-key-budget-ruin/HEAD) to run
the scripts online.

### Local Installation
If you want to run it locally on your machine, Python3 and Jupyter are needed.
The present code was developed and tested with the following versions:
- Python 3.10
- numpy 1.24
- scipy 1.10
- matplotlib 3.7

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file. 
You can then recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported by the German Research Foundation (DFG) under grant
BE 8098/1-1, by the German Federal Ministry of Education and Research (BMBF)
within the national initiative on 6G Communication Systems through the research
hub 6G-life under Grant 16KISK001K, and by the U.S National Science Foundation
under Grants CCF-1908308 and CNS-2128448.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@misc{Besser2023secretkeybudget,
  author = {Besser, Karl-Ludwig and Schaefer, Rafael F. and Poor, H. Vincent},
  title = {Reliability and Latency Analysis for Wireless Communication Systems with a Secret-Key Budget},
  year = {2023},
  month = {4},
  eprint = {2304.02538},
  archiveprefix = {arXiv},
  primaryclass = {cs.IT},
}
```
