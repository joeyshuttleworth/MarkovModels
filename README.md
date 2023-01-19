# Empirical quantification of predictive uncertainty due to model discrepancy by training with an ensemble of experimental designs: an application to hERG kinetics
This repository contains the code necessary to reproduce the Figures presented in our publication: Empirical quantification of predictive uncertainty due to model discrepancy by training with an ensemble of experimental designs: an application to hERG kinetic, Joseph G. Shuttleworth, Chon Lok Lei, Dominic G. Whittaker, Simon P. Preston and Gary R. Mirams.

This repository consists of package for the processing and analysis of patch-clamp electrophysiology data. Some of this functionality is used in the paper. The code ran to produce our Figures are the `scripts` directory, and the corresponding output is provided in the `output` directory.

## Installation

It is recommended to install libraries and run scripts in a virtual environment to avoid version conflicts between different projects. In order to do this, follow these steps:
- `virtualenv folder_name` or if you have both python 2 and 3: `virtualenv --python=python3 folder_name`. Should `virtualenv` not be recognised you may need to call it as `python -m virtualenv folder_name` or (`python -m virtualenv folder_name`). If that doesn't work you may need to install virtualenv first `pip install virtualenv`.
- Activate the virtual environment using `source folder_name/bin/activate`. Simply type `deactivate` to exit the virtual environment at the end of a session.
- Install [graphviz](https://graphviz.org/). On Ubuntu, this is done by running `sudo apt install graphviz graphviz-dev`. 
- Install the MarkovModels package by running `pip install -e .`.

## Scripts
Figure 1 was produced using `scripts/fix_wrong_param_study/simple_example.py`.

For Case I, the computations are performed using `scripts/fix_wrong_param_study/fix_wrong_params`. Then, Figure 4 is produced using `scripts/fix_wrong_param_study/big_multi` with these results.

The synthetic dataset used for Case II was produced using `scripts/fix_param_study/generate_synthetic_data.py` and this dataset was used to fit both the Wang and Beattie models using `scripts/fix_wrong_param_study/fit_all_wells_and_protocols.py`. These results are summarised using `scripts/fix_wrong_parma_study/CaseII_figure.py` and `error_compare_plot.py`.

The scripts used to produce each figure are shown in the following table:

| Figure  | script  |
| ------- | ------- |
|         |         |

## Running
To run a script execute it using Python. For example,
```python3  scripts/fix_wrong_param_study/fix_wrong_parameter.py --protocols sis staircase```

## Protocols
A list of voltage-clamp protocols are provided in  `/MarkovModels/protocols`. These are `.csv` files which describe time-series data. The filenames which correspond to the protocols used in the data are shown in the table below.

| protocol      | filename        |
| : ----------- | :-----------    |
| d1            | hhbrute3gstep'  |
| d2            | sis             |
| d3            | spacefill19     |
| d4            | staircaseramp1  |
| d5            | wangbrute3gstep |
| d^*           | longap          |

## Results
All of the computational results mentioned in the paper are provided in the `/results` directory. In `/results`, each subdirectory includes an `info.txt` file which lists the command run to produce the output.
