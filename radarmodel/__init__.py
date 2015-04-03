"""Radar modeling.

.. currentmodule:: radarmodel

Models
------

.. autosummary::
    :toctree:

    PointGrid

"""
from . import point
from . import pointgrid

from .pointgrid import PointGrid

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
