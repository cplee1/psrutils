from importlib.metadata import version

from matplotlib.pyplot import rcParams

from .cube import *
from .kernels import *
from .logger import *
from .misc import *
from .plotting import *
from .polarisation import *
from .profile import *
from .rm import *

__version__ = version(__name__)

rcParams["figure.dpi"] = 300
rcParams["font.size"] = 12
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["font.family"] = "serif"

# Uncomment only if LaTeX is installed
# rcParams["text.usetex"] = True
# rcParams["font.serif"] = "cm"

del version, rcParams
