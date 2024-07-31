from importlib import metadata

from .blocks import (
    Block,
    BlockFunction,
    CurveFit,
    Exponential,
    Periodic,
    Polynomial,
    plot_functions,
)

__all__ = [
    "Block",
    "BlockFunction",
    "CurveFit",
    "Periodic",
    "Polynomial",
    "Exponential",
    "plot_functions",
]

__version__ = metadata.version("genjax-blocks")
