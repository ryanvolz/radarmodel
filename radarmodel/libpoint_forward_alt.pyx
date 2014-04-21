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
cimport pyfftw.pyfftw as pyfftw

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
#     y[m] = \sum_{n,p} ( 1/sqrt(N) * e^{2*\pi*i*n*(R*m - p + L - 1)/N} 
#                        * s[R*m - p + L - 1] * x[n, p] )
#          = \sum_{n,l} ( 1/sqrt(N) * e^{2*\pi*i*n*l/N} 
#                        * s[l] * x[n, R*m - l + L - 1] )
# for a given N, R, s[k], and variable x[n, p]. The index n varies from 0 to
# N - 1, while p varies from 0 to R*M + L - R - 1.
# The 1/sqrt(N) term is included so that applying this model to the result of
# the adjoint operation (with same scaling) is well-scaled. In other words,
# the entries of A*Astar along the diagonal equal the norm of s (except
# for the first len(s) entries, which give the norm of the first entries
# of s).

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef freqcode(stype[::1] s_over_sqrtN, xytype[:, ::1] x_aligned, xytype[:, ::1] X, 
              pyfftw.FFTW ifft, Py_ssize_t M, Py_ssize_t R, xytype[:, ::1] x):
    cdef Py_ssize_t L = s_over_sqrtN.shape[0]
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t m, l
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
    # (s_over_sqrtN*Xstrided).sum(axis=1) :
    for m in prange(M, nogil=True):
        ym = 0
        # constraints on l from bounds of x:
        # Rm - l + L - 1 >= 0:
        #       l <= Rm + L - 1 < Rm + L
        # Rm - l + L - 1 <= RM + L - R - 1:
        #       l >= Rm - RM +  R
        # but constraints on l from bounds of s imply 0 <= l < L
        # so lstart = 0
        #    lstop = L
        for l in xrange(L):
            ym = ym + s_over_sqrtN[l]*X[l % N, R*m - l + L - 1] # no += because of prange
        y[m] = ym

    return y_ndarray

def FreqCodeCython(stype[::1] s, xytype[:, ::1] x_aligned, xytype[:, ::1] X, 
                   pyfftw.FFTW ifft, Py_ssize_t M, Py_ssize_t R=1):
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