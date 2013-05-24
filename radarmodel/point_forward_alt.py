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
import scipy.sparse as sparse
import pyfftw
import multiprocessing

import libpoint_forward_alt

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'FreqCodeStrided', 'FreqCodeCython']

# These Point forward models implement the equation:
#     y[m] = \sum_{n,p} 1/N * e^{2*\pi*i*n*(R*m - p)/N} * s[R*m - p] * x[n, p]
#          = \sum_{n,k} 1/N * e^{2*\pi*i*n*k/N} * s[k] * x[n, R*m - k]
# for a given N, R, s[k], and variable x[n, p].
# The 1/N term is included so that applying this model to the result of
# the adjoint operation (without scaling) is well-scaled. In other words,
# the entries of A*Astar along the diagonal equal the norm of s (except
# for the first len(s) entries, which give the norm of the first entries
# of s).

def DirectSum(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    def direct_sum(x):
        y = np.zeros(M, dtype=xydtype)
        for m in xrange(M):
            ym = 0
            for k in xrange(max(0, R*m - R*M + 1), min(L, R*m + 1)):
                s_k = s[k]
                for n in xrange(N):
                    ym += 1/N*np.exp(2*np.pi*1j*n*k/N)*s_k*x[n, R*m - k]
            y[m] = ym

        return y
    
    return direct_sum

# FreqCode implementations apply the Fourier frequency synthesis first by performing
# an IFFT, then proceed with code modulation

def FreqCodeStrided(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # when N < L, we need to wrap result of FFT so that we can get L samples
    # do this by upsampling x before taking FFT, which accomplishes this wrapping
    step = L // N + 1
    nfft = N*step
    
    # IFFT below from FFTW does not include 1/N factor, so include it in s
    s_over_N = s/N
    
    x_aligned = pyfftw.n_byte_align(np.zeros((R*M, nfft), xydtype), 16)
    
    align_byte = 16
    n_zeros = (L-1)*(nfft-1)
    Xmem = np.zeros((R*M*nfft + n_zeros)*xydtype.itemsize + align_byte, dtype='int8')
    # find where to start X in Xmem so that it is byte aligned to align_byte
    align_offset = (align_byte - Xmem[n_zeros*xydtype.itemsize:].ctypes.data) % align_byte
    X = np.frombuffer(Xmem[(n_zeros*xydtype.itemsize + 
                            align_offset):(-align_byte + align_offset)].data,
                      dtype=xydtype).reshape((R*M, nfft))
    
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD', threads=_THREADS)
    
    # Xstrided[m, l] = X[Rm - l, l] where negative indices do not wrap
    #
    # Xstrided = [  X[0, 0]          0        ...    0         0      ...          0
    #               X[R, 0]      X[R-1, 1]    ... X[0, R]      0      ...          0
    #               X[2R, 0]     X[2R-1, 1]   ... X[R, R] X[R-1, R+1] ...          0
    #                  :             :        ...    :         :      ...          :
    #             X[R(M-1), 0] X[R(M-1)-1, 1] ...   ...       ...     ... X[R(M-1)-L-1, L-1] ]
    
    # it is ok that these strides end up accessing negative indices wrt X b/c we allocated
    # the necessary zeros for Xmem above
    Xstrided = np.lib.stride_tricks.as_strided(X, (M, L),
                                               (R*X.strides[0], X.strides[1] - X.strides[0]))

    # so we don't have to allocate new memory every time we multiply s_over_N with Xstrided
    sX = np.zeros((M, L), xydtype)
        
    def freqcode_strided(x):
        x_aligned[:, ::step] = x.T # upsample along FFT axis by step, results in wrapped FFT
        ifft.execute() # input is x_aligned, output is X
        np.multiply(s_over_N, Xstrided, out=sX)
        y = sX.sum(axis=1)
        return y
    
    return freqcode_strided

def FreqCodeCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    x_aligned = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    X = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD', 
                       axes=(0,), threads=_THREADS)
    
    return libpoint_forward_alt.FreqCodeCython(s, x_aligned, X, ifft, M, R)