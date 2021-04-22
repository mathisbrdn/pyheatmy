"""
Module that implements a decorator allowing a bound method written in Python to become a "checker".
Also creates a new associated error type.
"""

from functools import wraps
from inspect import ismethod


class ComputationOrderException(Exception):
    """Exception raised when a method with a needed tag is computed before the linked checker."""


def checker(checked_meth):
    """
    Transform a bound method written in Python to a "checker" method.
    Each method decorated with the .needed would raise a ComputationOrderException.
    It is also possible to reset the checker with .reset.

    Args:
        checked_meth (method): a bound method written in Python

    Returns:
            method: checker method
    """

    assert ismethod(checked_meth), "checked_meth has to be a method"

    def reset(col):
        col.__dict__["_" + checked_meth.__name__] = False

    def needed(meth):
        @wraps(meth)
        def new_meth(self, *args, **kargs):
            if (
                "_" + checked_meth.__name__ in self.__dict__
                and self.__dict__["_" + checked_meth.__name__]
            ):
                return meth(self, *args, **kargs)
            raise ComputationOrderException(
                f"{checked_meth.__name__} has to be computed before calling {meth.__name__}."
            )

        return new_meth

    @wraps(checked_meth)
    def wrapper(self, *args, **kwargs):
        self.__dict__["_" + checked_meth.__name__] = True
        return checked_meth(self, *args, **kwargs)

    wrapper.reset = reset
    wrapper.needed = needed
    return wrapper


__all__ = ["checker"]
