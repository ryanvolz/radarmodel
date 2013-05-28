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

from __future__ import division
import numpy as np
import numba
from numba.decorators import jit, autojit
import scipy.sparse as sparse
import pyfftw
import multiprocessing

from radarmodel import libpoint_adjoint_alt

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'CodeFreqStrided', 'CodeFreqCython']

# These Point adjoint models implement the equation:
#     x[n, p] = \sum_m e^{-2*\pi*i*n*(R*m - p)/N} * s*[R*m - p] * y[m]
# for a given N, R, s*[k], and variable y[m].
#             = \sum_k e^{-2*\pi*i*n*k/N} * s*[k] * y_R[k + p]
# where y_R = upsampled y by R (insert R-1 zeros after each original element).
#
# This amounts to sweeping demodulation of the received signal using the complex
# conjugate of the transmitted waveform followed by calculation of the Fourier
# spectrum for segments of the received signal.
# The Fourier transform is taken with the signal delay removed.

def DirectSum(s, N, M, R=1):
    L = len(s)
    s_conj = s.conj()
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    def direct_sum(y):
        x = np.zeros((N, R*M), dtype=xydtype)
        for n in xrange(N):
            for p in xrange(R*M):
                xnp = 0
                # constraints on m from bounds of s:
                # Rm - p >= 0:
                #       m >= ceil(p/R) --> m >= floor((p - 1)/R) + 1
                # Rm - p <= L-1:
                #       m <= floor((p + L - 1)/R)
                for m in xrange((p - 1)//R + 1, min(M, (p + L - 1)//R + 1)):
                    # downshift modulate signal by frequency given by p
                    # then correlate with conjugate of transmitted signal
                    xnp += s_conj[R*m - p]*np.exp(-2*np.pi*1j*n*(R*m - p)/N)*y[m]
                x[n, p] = xnp

        return x
    
    return direct_sum

# CodeFreq implementations apply the code demodulation first, then the 
# Fourier frequency analysis via FFT

def CodeFreqStrided(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step
    
    s_conj = s.conj()
    
    # ypad is y upsampled by R, then padded with L-1 zeros at end:
    #         |<---R---->|            |<-----R---->| |<-(L-1)->|
    # ypad = [y[0] 0 ... 0 y[1] 0 ... y[M-1] 0 ... 0 0 0 ... 0 0]
    #         |<----------------RM---------------->|
    ypad = np.zeros(R*M + L - 1, xydtype)
    # yshifted[p, k] = ypad[k + p]
    yshifted = np.lib.stride_tricks.as_strided(ypad, (R*M, L), 
                                               (ypad.itemsize, ypad.itemsize))
    
    demodpad = pyfftw.n_byte_align(np.zeros((R*M, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)
    
    def codefreq_strided(y):
        ypad[:R*M:R] = y
        np.multiply(yshifted, s_conj, demodpad[:, :L])
        fft.execute() # input is demodpad, output is x_aligned
        x = np.array(x_aligned[:, ::step].T) # we need a copy, which np.array provides
        return x
    
    return codefreq_strided

def CodeFreqCython(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step
    
    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')
    
    demodpad = pyfftw.n_byte_align(np.zeros((R*M, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)
    
    #demodpad = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    #x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    #fft = pyfftw.FFTW(demodpad, x_aligned, axes=(0,), threads=_THREADS)
    
    return libpoint_adjoint_alt.CodeFreqCython(s, demodpad, x_aligned, fft, step, N, M, R)