import builtins
from typing import Any

import numpy as np

__all__ = ["pythonise"]


def pythonise(input: Any) -> Any:
    """Convert numpy types to builtin types using recursion.

    Parameters
    ----------
    input : `Any`
        A number, iterator, or dictionary.

    Returns
    -------
    output : `Any`
        A number, iterator, or dictionary containing only builtin types.
    """
    match type(input):
        case np.bool_:
            output = bool(input)
        case np.int_ | np.int32:
            output = int(input)
        case np.float_ | np.int32:
            output = float(input)
        case np.str_:
            output = str(input)
        case builtins.tuple:
            output = tuple(pythonise(item) for item in input)
        case builtins.list:
            output = [pythonise(item) for item in input]
        case builtins.dict:
            output = {key: pythonise(val) for (key, val) in input.items()}
        case np.ndarray:
            output = pythonise(input.tolist())
        case _:
            output = input
    return output
