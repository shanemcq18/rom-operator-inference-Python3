# __init__.py
"""Operator Inference for data-driven model reduction of dynamical systems.

Author: Shane A. McQuarrie, Karen Willcox, and OpInf Contributors
Maintainer: Shane A. McQuarrie
GitHub: https://github.com/operator-inference/opinf
"""

__version__ = "0.6.0"

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

from .roms import ROM, ParametricROM

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
    "ROM",
    "ParametricROM",
]
