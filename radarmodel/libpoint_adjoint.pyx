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
from cython cimport view
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

# These Point adjoint models implement the equation:
#     x[n, p] = \sum_m 1/sqrt(N) * e^{-2*\pi*i*n*m/N} * s*[R*m - p] * y[m]
# for a given N, R, s*[k], and variable y[m]. The index n varies from 0 to N-1, 
# while p varies from 0 to R*M + L - R - 1 to facilitate all pairings of s* and y.
#
# This amounts to sweeping demodulation of the received signal using the complex
# conjugate of the transmitted waveform followed by calculation of the Fourier
# spectrum for segments of the received signal.
# The Fourier transform is taken with the signal delay intact.
# The 1/sqrt(N) term is included so that applying the forward model (with same
# scaling) to the result of this adjoint operation is well-scaled. In other 
# words, the entries of A*Astar along the diagonal equal the norm of s (except
# for the first len(s) entries, which give the norm of the first entries
# of s).

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef direct_sum(stype[::1] s_conj, xytype[:, ::1] dftmat, Py_ssize_t R, xytype[::1] y):
    cdef Py_ssize_t L = s_conj.shape[0]
    cdef Py_ssize_t N = dftmat.shape[0]
    cdef Py_ssize_t M = dftmat.shape[1]
    cdef Py_ssize_t P = R*M + L - R
    cdef Py_ssize_t m, n, p, mstart, mstop
    cdef xytype xnp

    cdef np.ndarray x_ndarray
    cdef xytype[:, ::1] x
    cdef np.npy_intp *xshape = [N, P]
    if xytype is cython.floatcomplex:
        # we set every entry, so empty is ok
        x_ndarray = np.PyArray_EMPTY(2, xshape, np.NPY_COMPLEX64, 0)
    elif xytype is cython.doublecomplex:
        # we set every entry, so empty is ok
        x_ndarray = np.PyArray_EMPTY(2, xshape, np.NPY_COMPLEX128, 0)
    x = x_ndarray
    
    for p in prange(P, nogil=True):
        # constraints on m from bounds of s:
        # Rm - p + L - 1 >= 0:
        #       m >= ceil((p - L + 1)/R) --> m >= floor((p - L)/R) + 1
        # Rm - p + L - 1 <= L - 1:
        #       m <= floor(p/R)
        # add R before division so calculation of (p - L)//R + 1 <= 0 
        # when it should be with cdivision semantics (floor toward 0)
        mstart = max(0, (p - L + R)//R)
        mstop = min(M, p//R + 1)
        for n in xrange(N):
            xnp = 0
            for m in xrange(mstart, mstop):
                xnp = xnp + s_conj[R*m - p + L - 1]*dftmat[n, m]*y[m] # no += because of prange
            x[n, p] = xnp

    return x_ndarray

def DirectSumCython(stype[::1] s, xytype[:, ::1] dftmat, Py_ssize_t R=1):
    cdef xytype[:, ::1] dftmat2 = dftmat # work around closure scope bug which doesn't include fused arguments

    cdef stype[::1] s_conj = np.conj(s)

    if xytype is cython.floatcomplex:
        def direct_sum_cython(cython.floatcomplex[::1] y):
            return direct_sum(s_conj, dftmat2, R, y)
    elif xytype is cython.doublecomplex:
        def direct_sum_cython(cython.doublecomplex[::1] y):
            return direct_sum(s_conj, dftmat2, R, y)

    return direct_sum_cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef codefreq(stype[::1] s_conj_over_sqrtN, xytype[:, ::1] demodpad, xytype[:, ::1] x_aligned, 
              object fft, Py_ssize_t step, Py_ssize_t N, Py_ssize_t M, Py_ssize_t R,
              xytype[::1] y):
    cdef Py_ssize_t L = s_conj_over_sqrtN.shape[0]
    cdef Py_ssize_t P = R*M + L - R
    cdef Py_ssize_t m, p, mstart, mstop
    cdef xytype ym

    cdef np.ndarray x_ndarray
    cdef xytype[:, ::view.contiguous] x
    cdef np.npy_intp *xshape = [N, P]
    if xytype is cython.floatcomplex:
        # we set every entry, so empty is ok
        x_ndarray = np.PyArray_EMPTY(2, xshape, np.NPY_COMPLEX64, 0)
    elif xytype is cython.doublecomplex:
        # we set every entry, so empty is ok
        x_ndarray = np.PyArray_EMPTY(2, xshape, np.NPY_COMPLEX128, 0)
    x = x_ndarray

    # np.multiply(smatstar, y, demodpad[:, :M]) :
    for p in prange(P, nogil=True):
        # constraints on m from bounds of s:
        # Rm - p + L - 1 >= 0:
        #       m >= ceil((p - L + 1)/R) --> m >= floor((p - L)/R) + 1
        # Rm - p + L - 1 <= L - 1:
        #       m <= floor(p/R)
        # add R before division so calculation of (p - L)//R + 1 <= 0 
        # when it should be with cdivision semantics (floor toward 0)
        mstart = max(0, (p - L + R)//R)
        mstop = min(M, p//R + 1)
        for m in xrange(mstart, mstop):
            demodpad[p, m] = s_conj_over_sqrtN[R*m - p + L - 1]*y[m]

    fft.execute() # input is demodpad, output is x_aligned
    x[:, :] = x_aligned.T[::step, :]

    return x_ndarray

def CodeFreqCython(stype[::1] s, xytype[:, ::1] demodpad, xytype[:, ::1] x_aligned, 
                   object fft, Py_ssize_t step, Py_ssize_t N, Py_ssize_t M, Py_ssize_t R):
    cdef xytype[:, ::1] demodpad2 = demodpad # work around closure scope bug which doesn't include fused arguments
    cdef xytype[:, ::1] x_aligned2 = x_aligned # work around closure scope bug which doesn't include fused arguments

    cdef stype[::1] s_conj_over_sqrtN = np.conj(s)/np.sqrt(N)

    if xytype is cython.floatcomplex:
        def codefreq_cython(cython.floatcomplex[::1] y):
            return codefreq(s_conj_over_sqrtN, demodpad2, x_aligned2, fft, step, N, M, R, y)
    elif xytype is cython.doublecomplex:
        def codefreq_cython(cython.doublecomplex[::1] y):
            return codefreq(s_conj_over_sqrtN, demodpad2, x_aligned2, fft, step, N, M, R, y)

    return codefreq_cython