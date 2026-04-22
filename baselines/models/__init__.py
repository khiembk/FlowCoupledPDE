"""Baseline neural operator models for coupled PDE benchmarks."""

from .fno import FNO1d, FNO2d
from .ufno import UFNO1d, UFNO2d
from .deeponet import DeepONet1d, DeepONet2d
from .transolver import Transolver1d, Transolver2d
from .cmwno import CMWNO1d, CMWNO2d
from .compol import COMPOL1d, COMPOL2d
from .diffusion_pde import DiffusionPDE

__all__ = [
    "FNO1d", "FNO2d",
    "UFNO1d", "UFNO2d",
    "DeepONet1d", "DeepONet2d",
    "Transolver1d", "Transolver2d",
    "CMWNO1d", "CMWNO2d",
    "COMPOL1d", "COMPOL2d",
    "DiffusionPDE",
]
