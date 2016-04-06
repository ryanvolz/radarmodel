# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import itertools

from radarmodel.pointgrid import TxRef, RxRef

from radarmodel.util import get_random_uniform, get_random_oncircle

def txref_forward_reference(s, x, y):
    L = len(s)
    P = x.shape[0]
    N = x.shape[1]
    M = len(y)
    R = (P - L)//(M - 1)

    y[:] = 0

    for m in xrange(M):
        for l in xrange(L):
            p = R*m - l + L - 1
            for n in xrange(N):
                # difference from rxref is l in exponential instead of m
                y[m] += 1/np.sqrt(N)*np.exp(2*np.pi*1j*n*l/N)*s[l]*x[p, n]

    return y

def txref_adjoint_x_reference(y, s, x):
    M = len(y)
    L = len(s)
    P = x.shape[0]
    N = x.shape[1]
    R = (P - L)//(M - 1)

    sc = s.conj()

    x[...] = 0

    for p in xrange(P):
        for n in xrange(N):
            for m in xrange(M):
                l = R*m - p + L - 1
                if l < 0 or l >= L:
                    continue
                # difference from rxref is l in exponential instead of m
                x[p, n] += 1/np.sqrt(N)*np.exp(2*np.pi*1j*n*l/N)*sc[l]*y[m]

    return x

def rxref_forward_reference(s, x, y):
    L = len(s)
    P = x.shape[0]
    N = x.shape[1]
    M = len(y)
    R = (P - L)//(M - 1)

    y[:] = 0

    for m in xrange(M):
        for l in xrange(L):
            p = R*m - l + L - 1
            for n in xrange(N):
                # difference from txref is m in exponential instead of l
                y[m] += 1/np.sqrt(N)*np.exp(2*np.pi*1j*n*m/N)*s[l]*x[p, n]

    return y

def rxref_adjoint_x_reference(y, s, x):
    M = len(y)
    L = len(s)
    P = x.shape[0]
    N = x.shape[1]
    R = (P - L)//(M - 1)

    sc = s.conj()

    x[...] = 0

    for p in xrange(P):
        for n in xrange(N):
            for m in xrange(M):
                l = R*m - p + L - 1
                if l < 0 or l >= L:
                    continue
                # difference from txref is m in exponential instead of l
                x[p, n] += 1/np.sqrt(N)*np.exp(2*np.pi*1j*n*m/N)*sc[l]*y[m]

    return x

def check_equal_reference(fun, reffun, args):
    err_msg = 'Result of "{0}" (y) does not match reference "{1}" (x)'

    def call():
        y0 = reffun(*args)
        y = fun(*args)
        try:
            np.testing.assert_array_almost_equal_nulp(y0, y, nulp=10000)
        except AssertionError as e:
            e.args += (err_msg.format(fun.__name__, reffun.__name__),)
            raise

    call.description = 'test_pointgrid_equal_reference: {0}, '.format(
        reffun.__name__,
    )
    call.description += 'L={0}, M={1}, N={2}, R={3}, {4}'

    return call

def test_equal_reference():
    clss = (TxRef, RxRef)
    refs = (dict(forward=txref_forward_reference,
                 adjoint_x=txref_adjoint_x_reference,
            ),
            dict(forward=rxref_forward_reference,
                 adjoint_x=rxref_adjoint_x_reference,
            ),
           )
    Ls = (13, 13, 13)
    Ms = (37, 37, 10)
    Ns = (13, 64, 27)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)

    np.random.seed(1)

    for (cls, ref), (L, M, N), R, sdtype in itertools.product(
        zip(clss, refs), zip(Ls, Ms, Ns), Rs, sdtypes
    ):
        model = cls(L=L, M=M, N=N, R=R, precision=sdtype)

        # forward operation
        fun = model.forward
        reffun = ref['forward']
        s = get_random_oncircle(model.sshape, model.sdtype)
        x = get_random_uniform(model.xshape, model.xydtype)
        y = np.zeros(model.yshape, model.xydtype)
        callable_test = check_equal_reference(fun, reffun, (s, x, y))
        callable_test.description = callable_test.description.format(
            L, M, N, R, np.dtype(sdtype).str
        )
        yield callable_test

        # adjoint_x operation
        fun = model.adjoint_x
        reffun = ref['adjoint_x']
        y = get_random_uniform(model.yshape, model.xydtype)
        s = get_random_oncircle(model.sshape, model.sdtype)
        x = np.zeros(model.xshape, model.xydtype)
        callable_test = check_equal_reference(fun, reffun, (y, s, x))
        callable_test.description = callable_test.description.format(
            L, M, N, R, np.dtype(sdtype).str
        )
        yield callable_test

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture',
                         #'--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)
