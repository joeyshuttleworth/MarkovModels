import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
