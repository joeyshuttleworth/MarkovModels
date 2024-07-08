# Qaulity Control

Automated quality control for ion channel voltage clamp measurements.

### hERG CHO cells
- [hergqc.py](./hergqc.py): A module for QC (selection) for hERG measurement (currently for only either staircase protocol or staircase-ramp protocol).
- [leak\_fit.py](./leak_fit.py): A module containing utility functions and class for doing leak fitting and EK estimating (currenly for only staircase-ramp protocol).
- [fcap.py](./fcap.py): A module containing functions for removing capacitive spikes (only for QC purpose).

[leak\_fit.py](./leak_fit.py) requires [PINTS](https://github.com/pints-team/pints)


# Example

See [nanion-data-export](https://github.com/CardiacModelling/nanion-data-export) GitHub repository.
