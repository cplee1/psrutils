from importlib.metadata import version

from .constants import *
from .core import *
from .jit_kernels import *
from .logger import *
from .plotting import *
from .rm import *

__version__ = version(__name__)
