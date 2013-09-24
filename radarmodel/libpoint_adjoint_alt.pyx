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
from cython cimport view
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

# These Point adjoint models implement the equation:
#     x[n, p] = \sum_m 1/sqrt(N) * e^{-2*\pi*i*n*(R*m - p)/N} * s*[R*m - p] * y[m]
# for a given N, R, s*[k], and variable y[m].
#             = \sum_k 1/sqrt(N) * e^{-2*\pi*i*n*k/N} * s*[k] * y_R[k + p]
# where y_R = upsampled y by R (insert R-1 zeros after each original element).
#
# This amounts to sweeping demodulation of the received signal using the complex
# conjugate of the transmitted waveform followed by calculation of the Fourier
# spectrum for segments of the received signal.
# The Fourier transform is taken with the signal delay removed.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef codefreq(stype[::1] s_conj_over_sqrtN, xytype[:, ::1] demodpad, xytype[:, ::1] x_aligned, 
              pyfftw.FFTW fft, Py_ssize_t step, Py_ssize_t N, Py_ssize_t M, Py_ssize_t R,
              xytype[::1] y):
    cdef Py_ssize_t L = s_conj_over_sqrtN.shape[0]
    cdef Py_ssize_t p, m, k, mstart, mstop
    cdef xytype ym

    cdef np.ndarray x_ndarray
    cdef xytype[:, ::view.contiguous] x
    cdef np.npy_intp *xshape = [N, R*M]
    if xytype is cython.floatcomplex:
        # we set every entry, so empty is ok
        x_ndarray = np.PyArray_EMPTY(2, xshape, np.NPY_COMPLEX64, 0)
    elif xytype is cython.doublecomplex:
        # we set every entry, so empty is ok
        x_ndarray = np.PyArray_EMPTY(2, xshape, np.NPY_COMPLEX128, 0)
    x = x_ndarray

    # np.multiply(yshifted, s_conj, demodpad[:, :L]) :
    for p in prange(R*M, nogil=True):
        # constraints on m from bounds of s:
        # Rm - p >= 0:
        #       m >= ceil(p/R) --> m >= floor((p - 1)/R) + 1
        # Rm - p <= L-1:
        #       m <= floor((p + L - 1)/R)
        mstart = (p - 1 + R)//R # add R before division to guarantee numerator is positive
        mstop = min(M, (p + L - 1)//R + 1)
        for m in xrange(mstart, mstop):
            k = R*m - p
            demodpad[p, k] = s_conj_over_sqrtN[k]*y[m]

    #cdef Py_ssize_t kstart, kstop
    ## np.multiply(yshifted, s_conj, demodpad[:, :L]) :
    #for p in prange(R*M, nogil=True):
        #kstart = (R*M - p) % R # so k + p is always a multiple of R
        #kstop = min(L, R*M - p - R + 1)
        #for k from kstart <= k < kstop by R: #for k in range(kstart, kstop, R):
            #demodpad[p, k] = s_conj_over_sqrtN[k]*y[(k + p)//R]

    fft.execute() # input is demodpad, output is x_aligned
    x[:, :] = x_aligned.T[::step, :]

    return x_ndarray

def CodeFreqCython(stype[::1] s, xytype[:, ::1] demodpad, xytype[:, ::1] x_aligned, 
                   pyfftw.FFTW fft, Py_ssize_t step, Py_ssize_t N, Py_ssize_t M, Py_ssize_t R):
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