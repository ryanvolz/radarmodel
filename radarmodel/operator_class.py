# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Linear operator class.

.. currentmodule:: radarmodel.operator_class

.. autosummary::
    :toctree:

    LinearOperator

"""

import numpy as np

__all__ = [
    'LinearOperator',
]

class LinearOperator(object):
    """Linear operator A.

    This class defines a linear operator through its forward and adjoint
    operations. The forward operation is called using the `forward` method,
    while the adjoint operation is called using the `adjoint` method.

    For specifying a particular operator, inherit from this class and override
    the `_forward` and `_adjoint` methods.

    .. automethod:: __init__


    Attributes
    ----------

    inshape, outshape : tuple
        Tuples giving the shape of the input and output arrays of the forward
        operation (output and input arrays of the adjoint operation),
        respectively.

    indtype, outdtype : dtype
        Dtypes of the input and output arrays of the forward operation
        (output and input arrays of the adjoint operation), respectively.

    """
    def __init__(self, inshape, indtype, outshape, outdtype):
        """Create a linear operator for the given input and output specs.


        Parameters
        ----------

        inshape, outshape : tuple
            Tuples giving the shape of the input and output arrays of the
            forward operation (output and input arrays of the adjoint
            operation), respectively.

        indtype, outdtype : dtype
            Dtypes of the input and output arrays of the forward operation
            (output and input arrays of the adjoint operation), respectively.

        """
        self.inshape = inshape
        self.indtype = indtype
        self.outshape = outshape
        self.outdtype = outdtype

    def _forward(self, x, out):
        raise NotImplementedError

    def _adjoint(self, y, out):
        raise NotImplementedError

    def forward(self, x, out=None):
        """Calculate the forward operation, A(x).


        Parameters
        ----------

        x : ndarray of shape `inshape` and dtype `indtype`
            Input to forward operation A.

        out : ndarray of shape `outshape` and dtype `outdtype`
            Optional pre-allocated array for storing the output.


        Returns
        -------

        out : ndarray of shape `outshape` and dtype `outdtype`
            Output of the forward operation A.

        """
        if out is None:
            out = np.empty(self.outshape, self.outdtype)
        return self._forward(x, out)

    def adjoint(self, y, out=None):
        """Calculate the adjoint operation, A*(y).


        Parameters
        ----------

        y : ndarray of shape `outshape` and dtype `outdtype`
            Input to adjoint operation A*.

        out : ndarray of shape `outshape` and dtype `outdtype`
            Optional pre-allocated array for storing the output.


        Returns
        -------

        out : ndarray of shape `inshape` and dtype `indtype`
            Output of the adjoint operation A*.

        """
        if out is None:
            out = np.empty(self.inshape, self.indtype)
        return self._adjoint(y, out)

    # short aliases
    A = forward
    As = adjoint
