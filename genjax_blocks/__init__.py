from importlib import metadata

from .blocks import (
    Block,
    BlockFunction,
    CurveFit,
    Exponential,
    Periodic,
    Polynomial,
    plot_functions,
    DataModel,
    NoisyData,
    NoisyOutliersData,
    CurveDataModel,
)

__all__ = [
    "Block",
    "BlockFunction",
    "CurveFit",
    "Periodic",
    "Polynomial",
    "Exponential",
    "plot_functions",
    "DataModel",
    "NoisyData",
    "NoisyOutlierData",
    "CurveDataModel",
]

__version__ = metadata.version("genjax-blocks")
