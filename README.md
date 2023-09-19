# MarkovModels
## Installation

It is recommended to install libraries and run scripts in a virtual environment to avoid version conflicts between different projects. In order to do this, follow these steps:
- `virtualenv folder_name` or if you have both python 2 and 3: `virtualenv --python=python3 folder_name`. Should `virtualenv` not be recognised you may need to call it as `python -m virtualenv folder_name` or (`python -m virtualenv folder_name`). If that doesn't work you may need to install virtualenv first `pip install virtualenv`.
- Activate the virtual environment using `source folder_name/bin/activate`. Simply type `deactivate` to exit the virtual environment at the end of a session.
- Install [graphviz](https://graphviz.org/). On Ubuntu, this is done by running `sudo apt install graphviz graphviz-dev`. 
- Install gcc and build essential: `sudo apt-get install gcc build-essential`
- Install cmake: `sudo apt-get install cmake`
- Install scikit-build: `pip install scikit-build`
- Install the markovmodels package by running `pip install -e .`.


## Running
To run a script execute it using Python. For example,
```python3  scripts/fix_wrong_param_study/fix_wrong_parameter.py --protocols sis staircase```

