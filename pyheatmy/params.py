from collections import namedtuple
from dataclasses import dataclass
from random import uniform, gauss

from .utils import PARAM_LIST

Param = namedtuple("Parametres", PARAM_LIST)

@dataclass
class Carac:
    range: tuple
    sigma: float

    def perturb(self, val, /):
        new_val = val + gauss(0, self.sigma)
        while new_val > self.range[1]:
            new_val -= self.range[1]-self.range[0]
        while new_val < self.range[0]:
            new_val += self.range[1]-self.range[0]
        return new_val

class ParamsCaracs:
    def __init__(self, caracs):
        self.caracs = caracs

    def sample_params(self):
        return Param(*(
            uniform(*carac.range)
            for carac in self.caracs
        ))

    def perturb(self, param):
        return Param(*(
            carac.perturb(val)
            for carac, val in zip(self.caracs, param)
        ))

__all__ = [
    "Param",
    "Carac",
    "ParamsCaracs",
]