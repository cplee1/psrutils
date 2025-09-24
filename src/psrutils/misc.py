########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from typing import Any

import numpy as np
from psrqpy import QueryATNF

__all__ = ["pythonise"]


def pythonise(var: Any) -> Any:
    """Convert numpy types to builtin types.

    Parameters
    ----------
    var : Any
        A number or iterator.

    Returns
    -------
    Any
        The input variable cast into builtin types.
    """
    match var:
        case np.bool_():
            output = bool(var)
        case np.integer():
            output = int(var)
        case np.floating():
            output = float(var)
        case np.str_():
            output = str(var)
        case tuple():
            output = tuple(pythonise(item) for item in var)
        case list():
            output = [pythonise(item) for item in var]
        case dict():
            output = {key: pythonise(val) for (key, val) in var.items()}
        case np.ndarray():
            output = pythonise(var.tolist())
        case _:
            output = var
    return output


def jname_to_name(jname: str) -> str:
    if jname.startswith("B"):
        # Already a B-name
        return jname
    cat_table = QueryATNF(params=["PSRJ", "NAME"]).table
    cat_table.add_index("PSRJ")
    return str(cat_table.loc[jname]["NAME"])
