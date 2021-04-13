from dataclasses import dataclass

from numpy import array

from .params import Param

@dataclass
class State:
    params: Param
    energy: float
    ratio_accept: float
    temps: array
    flows: array

__all__ = ["State"]