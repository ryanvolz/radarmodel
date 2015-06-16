"""Mathematical radar models useful for inverting radar measurements.

.. currentmodule:: radarmodel

Models
------

.. autosummary::
    :toctree:

    pointgrid

Other
-----

.. autosummary::
    :toctree:

    delay_multiply
    time_varying_conv
    operator_class

"""
from . import pointgrid

from .pointgrid import PointGrid

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
