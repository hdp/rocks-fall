"""Rocks Fall: dice probability calculator"""

__version__ = "0.0.2"

from .dice import Builder, explode


__all__ = ["d", "explode"]


d = Builder()
