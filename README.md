# Python implementation of hERG ion channel model
This repository contains scripts for solving a hERG ion channel model analytically and using an ODE solver.

## Installation

It is recommended to install libraries and run scripts in a virtual environment to avoid version conflicts between different projects. In order to do this, follow these steps:
- `virtualenv folder_name` or if you have both python 2 and 3: `virtualenv --python=python3 folder_name`. Should `virtualenv` not be recognised you may need to call it as `python -m virtualenv folder_name` or (`python -m virtualenv folder_name`). If that doesn't work you may need to install virtualenv first `pip install virtualenv`.
- Activate the virtual environment using `source folder_name/bin/activate`. Simply type `deactivate` to exit the virtual environment at the end of a session.
- Install the required packages by typing `pip install -r requirements.txt`

## Running

[Joseph to Do]

## Tests

Tests are available to check whether the analytical solution of the ion channel model matches that obtained using an ODE solver. In order to run the tests, simply type `pytest test_hh_markov_analytic.py`.