from importlib import metadata

from .blocks import (
    Block,
    BlockFunction,
    CurveFit,
    Exponential,
    Periodic,
    Polynomial,
)

__all__ = [
    "Block",
    "BlockFunction",
    "CurveFit",
    "Periodic",
    "Polynomial",
    "Exponential",
]

__version__ = metadata.version("genjax-blocks")
