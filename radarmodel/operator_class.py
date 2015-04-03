# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

__all__ = [
    'LinearOperator',
]

class LinearOperator(object):
    def __init__(self, inshape, indtype, outshape, outdtype):
        self.inshape = inshape
        self.indtype = indtype
        self.outshape = outshape
        self.outdtype = outdtype

    def _forward(self, x, out):
        raise NotImplementedError

    def _adjoint(self, y, out):
        raise NotImplementedError

    def forward(self, x, out=None):
        if out is None:
            out = np.empty(self.outshape, self.outdtype)
        return self._forward(x, out)

    def adjoint(self, y, out=None):
        if out is None:
            out = np.empty(self.inshape, self.indtype)
        return self._adjoint(y, out)

    # short aliases
    A = forward
    As = adjoint
