# Copyright 2013 Ryan Volz

# This file is part of radarmodel.

# Radarmodel is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Radarmodel is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with radarmodel.  If not, see <http://www.gnu.org/licenses/>.


#cython: embedsignature=True

from __future__ import division
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport pyfftw

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
#     y[m] = \sum_{n,p} 1/sqrt(N) * e^{2*\pi*i*n*(R*m - p)/N} * s[R*m - p] * x[n, p]
# for a given N, R, s[k], and variable x[n, p].
# The 1/N term is included so that applying this model to the result of
# the adjoint operation (without scaling) is well-scaled. In other words,
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
    cdef Py_ssize_t m, l, lstart, lstop
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
        lstart = max(0, R*m - R*M + 1)
        lstop = min(L, R*m + 1)
        for l in xrange(lstart, lstop):
            ym = ym + s_over_sqrtN[l]*X[l % N, R*m - l] # no += because of prange
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