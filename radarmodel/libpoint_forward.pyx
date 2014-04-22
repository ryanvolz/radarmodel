#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

#cython: embedsignature=True

from __future__ import division
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

np.import_array() # or else we get segfaults when calling numpy C-api

ctypedef fused stype:
    cython.float
    cython.double
    cython.floatcomplex
    cython.doublecomplex

ctypedef fused xytype:
    cython.floatcomplex
    cython.doublecomplex

# These Point forward models implement the equation:
#     y[m] = \sum_{n,p} 1/sqrt(N) * e^{2*\pi*i*n*m/N} * s[R*m - p + L - 1] * x[n, p]
# for a given N, R, s[k], and variable x[n, p]. The index n varies from 0 to N - 1, 
# while p varies from 0 to R*M + L - R - 1.
# The 1/sqrt(N) term is included so that applying this model to the result of
# the adjoint operation (with same scaling) is well-scaled. In other words,
# the entries of A*Astar along the diagonal equal the norm of s (except
# for the first len(s) entries, which give the norm of the first entries
# of s).

@cython.boundscheck(False)
@cython.wraparound(False)
cdef direct_sum(stype[::1] s, xytype[:, ::1] idftmat, Py_ssize_t R, xytype[:, ::1] x):
    cdef Py_ssize_t L = s.shape[0]
    cdef Py_ssize_t M = idftmat.shape[0]
    cdef Py_ssize_t N = idftmat.shape[1]
    cdef Py_ssize_t P = x.shape[1]
    cdef Py_ssize_t m, n, p
    cdef xytype ym
    cdef stype sk

    cdef np.ndarray y_ndarray
    cdef xytype[::1] y
    if xytype is cython.floatcomplex:
        # we set every entry, so empty is ok
        y_ndarray = np.PyArray_EMPTY(1, <np.npy_intp*>&M, np.NPY_COMPLEX64, 0)
    elif xytype is cython.doublecomplex:
        # we set every entry, so empty is ok
        y_ndarray = np.PyArray_EMPTY(1, <np.npy_intp*>&M, np.NPY_COMPLEX128, 0)
    y = y_ndarray

    for m in prange(M, nogil=True):
        ym = 0
        # constraints on p from bounds of s:
        # Rm - p + L - 1 >= 0:
        #       p <= Rm + L - 1 < Rm + L
        # Rm - p + L - 1 <= L - 1:
        #       p >= Rm
        # but constraints on p from bounds of x imply 0 <= p < P = R*M + L - R
        # so pstart = max(0, R*m) = R*m
        #    pstop = min(P, R*m + L) = min(R*M + L - R, R*m + L) = R*m + L
        for p in xrange(R*m, R*m + L):
            sk = s[R*m - p + L - 1]
            for n in xrange(N):
                ym = ym + sk*idftmat[m, n]*x[n, p]
        y[m] = ym

    return y_ndarray

def DirectSumCython(stype[::1] s, xytype[:, ::1] idftmat, Py_ssize_t R=1):
    cdef stype[::1] s2 = s # work around closure scope bug which doesn't include fused arguments
    cdef xytype[:, ::1] idftmat2 = idftmat # work around closure scope bug which doesn't include fused arguments

    if xytype is cython.floatcomplex:
        def direct_sum_cython(cython.floatcomplex[:, ::1] x):
            return direct_sum(s2, idftmat2, R, x)
    elif xytype is cython.doublecomplex:
        def direct_sum_cython(cython.doublecomplex[:, ::1] x):
            return direct_sum(s2, idftmat2, R, x)

    return direct_sum_cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef freqcode(stype[::1] s_over_sqrtN, xytype[:, ::1] x_aligned, xytype[:, ::1] X, 
              object ifft, Py_ssize_t M, Py_ssize_t R, xytype[:, ::1] x):
    cdef Py_ssize_t L = s_over_sqrtN.shape[0]
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t P = X.shape[1]
    cdef Py_ssize_t m, p, pstart, pstop, m_mod_N
    cdef xytype ym

    cdef np.ndarray y_ndarray
    cdef xytype[::1] y
    if xytype is cython.floatcomplex:
        # we set every entry, so empty is ok
        y_ndarray = np.PyArray_EMPTY(1, <np.npy_intp*>&M, np.NPY_COMPLEX64, 0)
    elif xytype is cython.doublecomplex:
        # we set every entry, so empty is ok
        y_ndarray = np.PyArray_EMPTY(1, <np.npy_intp*>&M, np.NPY_COMPLEX128, 0)
    y = y_ndarray

    x_aligned[:, :] = x
    ifft.execute() # input is x_aligned, output is X
    # (smat*X).sum(axis=1) :
    for m in prange(M, nogil=True):
        ym = 0
        m_mod_N = m % N
        # constraints on p from bounds of s:
        # Rm - p + L - 1 >= 0:
        #       p <= Rm + L - 1 < Rm + L
        # Rm - p + L - 1 <= L - 1:
        #       p >= Rm
        # but constraints on p from bounds of x imply 0 <= p < P = R*M + L - R
        # so pstart = max(0, R*m) = R*m
        #    pstop = min(P, R*m + L) = min(R*M + L - R, R*m + L) = R*m + L
        for p in xrange(R*m, R*m + L):
            ym = ym + s_over_sqrtN[R*m - p + L - 1]*X[m_mod_N, p] # no += because of prange
        y[m] = ym

    return y_ndarray

def FreqCodeCython(stype[::1] s, xytype[:, ::1] x_aligned, xytype[:, ::1] X, 
                   object ifft, Py_ssize_t M, Py_ssize_t R=1):
    cdef xytype[:, ::1] x_aligned2 = x_aligned # work around closure scope bug which doesn't include fused arguments
    cdef xytype[:, ::1] X2 = X # work around closure scope bug which doesn't include fused arguments

    N = X.shape[0]
    cdef stype[::1] s_over_sqrtN = np.asarray(s)/np.sqrt(N)

    if xytype is cython.floatcomplex:
        def freqcode_cython(cython.floatcomplex[:, ::1] x):
            return freqcode(s_over_sqrtN, x_aligned2, X2, ifft, M, R, x)
    elif xytype is cython.doublecomplex:
        def freqcode_cython(cython.doublecomplex[:, ::1] x):
            return freqcode(s_over_sqrtN, x_aligned2, X2, ifft, M, R, x)

    return freqcode_cython