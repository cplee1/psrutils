# psrutils
A collection of Python-based utilities to analyse pulsar data.

# Installation
This package requires Python 3.10+ and PSRCHIVE with Python bindings
installed. All of the required dependencies can be installed using `conda`.
For example:

```bash
conda create -n psrutils python=3.10
conda activate psrutils
conda install conda-forge::dspsr
pip install git+https://github.com/cplee1/psrutils.git
```

# Credit
This repository is open source under the AFL-v3.0 license. A paper
describing the RM synthesis code is currently under review, and another one
describing the profile analysis code is in preparation. Until these papers
are published, please give credit by linking to this repository.