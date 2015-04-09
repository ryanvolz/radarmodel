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

def opnorm(op, reltol=1e-8, abstol=1e-6, maxits=100, printrate=None):
    """Estimate the l2-induced operator norm: sup_v ||A(v)||/||v|| for v != 0.

    Uses the power iteration method to estimate the operator norm of
    op.A and op.As.

    The type and shape of the input to A are specified by inshape and indtype.

    Returns a tuple: (norm of A, norm of As, vector inducing maximum scaling).
    """
    inshape = op.inshape
    indtype = op.indtype
    v0 = get_random_normal(inshape, indtype)
    v = v0/np.linalg.norm(v0)
    norm_f0 = 1
    norm_a0 = 1
    for k in xrange(maxits):
        Av = op.A(v)
        norm_f = np.linalg.norm(Av)
        w = Av/norm_f
        Asw = op.As(w)
        norm_a = np.linalg.norm(Asw)
        v = Asw/norm_a

        delta_f = abs(norm_f - norm_f0)
        delta_a = abs(norm_a - norm_a0)

        if printrate is not None and (k % printrate) == 0:
            print('Iteration {0}, forward norm: {1}, '
                  + 'adjoint norm: {2}'.format(k, norm_f, norm_a)
            )
        if (delta_f < abstol + reltol*max(norm_f, norm_f0)
            and delta_a < abstol + reltol*max(norm_a, norm_a0)):
            break

        norm_f0 = norm_f
        norm_a0 = norm_a

    return norm_f, norm_a, v

def check_opnorm(cls, L, M, N, R, sdtype):
    s = get_random_oncircle((L,), sdtype)
    s = s/np.linalg.norm(s)

    model = cls(L=L, M=M, N=N, R=R, precision=sdtype)
    op = model(s=s)

    true_Anorm = 1
    true_Asnorm = 1

    err_msg = 'Estimated {0} norm ({1}) does not match true {0} norm ({2})'

    def call():
        Anorm, Asnorm, v = opnorm(op, reltol=1e-10, abstol=1e-8, maxits=100)
        np.testing.assert_allclose(Anorm, true_Anorm, rtol=1e-4, atol=1e-2,
                          err_msg=err_msg.format('forward', Anorm, true_Anorm))
        np.testing.assert_allclose(Asnorm, true_Asnorm, rtol=1e-4, atol=1e-2,
                        err_msg=err_msg.format('adjoint', Asnorm, true_Asnorm))

    call.description = '{6}: s={0}({1}), x={2}({3}), N={4}, R={5}'.format(
        np.dtype(sdtype).str, L, np.dtype(op.indtype).str, M, N, R,
        cls.__name__,
    )

    return call

def test_opnorm():
    clss = (TxRef, RxRef)
    Ls = (13, 13, 13)
    Ms = (37, 37, 10)
    Ns = (13, 64, 27)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)

    np.random.seed(1)

    for cls, (L, M, N), R, sdtype in itertools.product(
        clss, zip(Ls, Ms, Ns), Rs, sdtypes
    ):
        callable_test = check_opnorm(cls, L, M, N, R, sdtype)
        callable_test.description = 'test_pointgrid_opnorm: '\
                                        + callable_test.description
        yield callable_test

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture',
                         #'--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)
