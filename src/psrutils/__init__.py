from importlib.metadata import version

from .core import *
from .jit_kernels import *
from .logger import *

__version__ = version(__name__)

# Speed of light [m/s]
C0 = 2.998e8

# The imaginary number
IMAG = complex(0.0, 1.0)
