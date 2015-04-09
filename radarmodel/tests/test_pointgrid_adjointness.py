# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import itertools

from radarmodel.pointgrid import TxRef, RxRef

from radarmodel.util import get_random_normal, get_random_oncircle

def adjointness_error(op, its=100):
    """Check adjointness of op.A and op.As for 'its' instances of random data.

    For random unit-normed x and y, this finds the error in the adjoint
    identity <Ax, y> == <x, A*y>:
        err = abs( vdot(A(x), y) - vdot(x, Astar(y)) ).

    The type and shape of the input to A are specified by inshape and indtype.

    Returns a vector of the error magnitudes.

    """
    inshape = op.inshape
    indtype = op.indtype
    outshape = op.outshape
    outdtype = op.outdtype

    x = get_random_normal(inshape, indtype)

    errs = np.zeros(its, dtype=indtype)
    for k in xrange(its):
        x = get_random_normal(inshape, indtype)
        x = x/np.linalg.norm(x)
        y = get_random_normal(outshape, outdtype)
        y = y/np.linalg.norm(y)
        ip_A = np.vdot(op.A(x), y)
        ip_Astar = np.vdot(x, op.As(y))
        errs[k] = np.abs(ip_A - ip_Astar)

    return errs

def check_adjointness(cls, L, N, M, R, sdtype):
    s = get_random_oncircle((L,), sdtype)
    s = s/np.linalg.norm(s)

    model = cls(L=L, M=M, N=N, R=R, precision=sdtype)
    op = model(s=s)

    err_msg = '{0} and {1} are not adjoints, with max error of {2}'

    def call():
        errs = adjointness_error(op, its=100)
        np.testing.assert_array_almost_equal(
            errs, 0, err_msg=err_msg.format(
                op.A.__name__, op.As.__name__, np.max(np.abs(errs)),
            )
        )

    call.description = '{6}: s={0}({1}), x={2}({3}), N={4}, R={5}'.format(
        np.dtype(sdtype).str, L, np.dtype(op.indtype).str, M, N, R,
        cls.__name__,
    )

    return call

def test_pointgrid_adjointness():
    clss = (TxRef, RxRef)
    Ls = (13, 13, 13)
    Ns = (13, 64, 27)
    Ms = (37, 37, 10)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)

    np.random.seed(1)

    for cls, (L, N, M), R, sdtype in itertools.product(
        clss, zip(Ls, Ns, Ms), Rs, sdtypes
    ):
        callable_test = check_adjointness(cls, L, N, M, R, sdtype)
        callable_test.description = 'test_pointgrid_adjointness: '\
                                        + callable_test.description
        yield callable_test

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture',
                         #'--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)
