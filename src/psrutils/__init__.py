from importlib.metadata import version

from matplotlib.pyplot import rcParams

from .constants import *
from .core import *
from .jit_kernels import *
from .logger import *
from .plotting import *
from .profile import *
from .rm import *

__version__ = version(__name__)


rcParams["font.size"] = 12
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["font.family"] = "serif"
# rcParams["text.usetex"] = True
# rcParams["font.serif"] = "cm"
