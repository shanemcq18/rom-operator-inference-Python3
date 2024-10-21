# __init__.py
"""Operator Inference for data-driven model reduction of dynamical systems.

Author: Willcox Research Group
Maintainer: Shane A. McQuarrie
GitHub:
    https://github.com/Willcox-Research-Group/rom-operator-inference-Python3
"""

__version__ = "0.5.9"

from . import (
    basis,
    errors,
    ddt,
    lift,
    lstsq,
    models,
    operators,
    pre,
    post,
    roms,
    utils,
)

from .roms import *

__all__ = [
    "basis",
    "errors",
    "ddt",
    "lift",
    "lstsq",
    "models",
    "operators",
    "pre",
    "post",
    "roms",
    "utils",
]

__all__ += roms.__all__