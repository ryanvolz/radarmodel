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

import libpoint_adjoint

_THREADS = multiprocessing.cpu_count()

# These Point adjoint models implement the equation:
#     x[n, p] = \sum_m e^{2*\pi*i*n*m/N} * s*[R*m - p] * y[m]
# for a given N, R, s*[k], and variable y[m].

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
                    xnp += s_conj[R*m - p]*np.exp(-2*np.pi*1j*n*m/N)*y[m]
                x[n, p] = xnp

        return x
    
    return direct_sum

def DirectSumNumba(s, N, M, R=1):
    L = len(s)
    s_conj = s.conj()
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    stype = numba.__getattribute__(str(s.dtype))
    xytype = numba.__getattribute__(str(xydtype))
    
    dftmat = np.exp(-2*np.pi*1j*np.arange(M)*np.arange(N)[:, None]/N)
    dftmat = dftmat.astype(xydtype) # use precision of output
    
    @jit(restype=xytype[:, ::1], 
         argtypes=[stype[::1], xytype[:, ::1], xytype[::1]],
         locals=dict(xnp=xytype))
    def direct_sum(s_conj, dftmat, y):
        x = np.zeros((N, R*M), dtype=y.dtype)
        for n in range(N):
            for p in range(R*M):
                xnp = 0
                # constraints on m from bounds of s:
                # Rm - p >= 0:
                #       m >= ceil(p/R) --> m >= floor((p - 1)/R) + 1
                # Rm - p <= L-1:
                #       m <= floor((p + L - 1)/R)
                for m in range((p - 1)//R + 1, min(M, (p + L - 1)//R + 1)):
                    # downshift modulate signal by frequency given by p
                    # then correlate with conjugate of transmitted signal
                    xnp += s_conj[R*m - p]*dftmat[n, m]*y[m]
                x[n, p] = xnp
        return x
    
    def direct_sum_numba(y):
        return direct_sum(s_conj, dftmat, y)
    
    return direct_sum_numba

def DirectSumCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    dftmat = np.exp(-2*np.pi*1j*np.arange(M)*np.arange(N)[:, None]/N)
    dftmat = dftmat.astype(xydtype) # use precision of output
    
    return libpoint_adjoint.DirectSumCython(s, dftmat, R)

# FreqCode implementations apply the Fourier frequency analysis first by performing
# a frequency downshift, then proceed with code correlation

def FreqCodeSparse(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    #         |<--(RM-1)-->| |<-----------L----------->| |<--(RM-L)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    #         |<-----RM------>|<-----------------RM------------------>|
    spad = np.hstack((np.zeros(R*M - 1, s.dtype), s, np.zeros(R*M - L, s.dtype)))
    
    # smatconj = 
    # [s*[0]     0     ...   0      0    ...   0   0 ... 0    0    ...   0     0   0 ... 0
    #  s*[R]  s*[R-1]  ... s*[0]    0    ...   0   0 ... 0    0    ...   0     0   0 ... 0
    #  s*[2R] s*[2R-1] ... s*[R] s*[R-1] ... s*[0] 0 ... 0    0    ...   0     0   0 ... 0
    #    :       :           :      :          :   :     :    :          :     :   : ... :   
    #    0       0     ...   0      0    ...   0   0 ... 0 s*[L-1] ... s*[1] s*[0] 0 ... 0 ]
    #                                                                              |<--->|
    #                                                                                R-1
    smatconj = np.lib.stride_tricks.as_strided(spad.conj()[(R*M - 1):], (M, R*M),
                                               (R*spad.itemsize, -spad.itemsize))
    smatconj = sparse.csr_matrix(smatconj)
    
    dftmat = np.exp(-2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(M)/N)
    dftmat = dftmat.astype(xydtype) # use precision of output
    
    def freqcode_sparse(y):
        return np.asarray((dftmat*y)*smatconj)
    
    return freqcode_sparse

# CodeFreq implementations apply the code demodulation first, then the 
# Fourier frequency analysis via FFT

def CodeFreqStrided(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # when N < M, still need to take FFT with nfft >= M so we don't lose data
    # then subsample to get our N points that we desire
    step = M // N + 1
    nfft = N*step
    
    #         |<--(RM-1)-->| |<-----------L----------->| |<--(RM-L)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    #         |<-----RM------>|<-----------------RM------------------>|
    spad = np.hstack((np.zeros(R*M - 1, s.dtype), s, np.zeros(R*M - L, s.dtype)))
    
    # smatstar = [s*[0]  s*[R]   s*[2R]  ...    0
    #               0   s*[R-1] s*[2R-1]        0
    #               :      :       :            :
    #               0    s*[0]   s*[R]   ...    0
    #               0      0    s*[R-1]  ...    0
    #               :      :       :            :
    #               0      0     s*[0]   ...    0
    #               0      0       0     ...    0
    #               :      :       :            :
    #               0      0       0     ...    0
    #               0      0       0     ... s*[L-1]
    #               :      :       :            :
    #               0      0       0     ...  s*[1]
    #               0      0       0     ...  s*[0]  
    #               0      0       0     ...    0     ---
    #               :      :       :            :      | R-1
    #               0      0       0     ...    0   ] ---
    smatstar = np.lib.stride_tricks.as_strided(spad.conj()[(R*M - 1):], (R*M, M),
                                               (-spad.itemsize, R*spad.itemsize))
    
    demodpad = pyfftw.n_byte_align(np.zeros((R*M, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)
    
    def codefreq_strided(y):
        np.multiply(smatstar, y, demodpad[:, :M])
        fft.execute() # input is demodpad, output is x_aligned
        x = np.array(x_aligned[:, ::step].T) # need a copy, which np.array provides
        return x
    
    return codefreq_strided

def CodeFreqCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # when N < M, still need to take FFT with nfft >= M so we don't lose data
    # then subsample to get our N points that we desire
    step = M // N + 1
    nfft = N*step
    
    demodpad = pyfftw.n_byte_align(np.zeros((R*M, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)
    
    return libpoint_adjoint.CodeFreqCython(s, demodpad, x_aligned, fft, step, N, M, R)

def CodeFreq2Strided(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    s_conj = s.conj()
    
    ypad = np.zeros(R*M + L - 1, xydtype)
    # yshifted[p, k] = ypad[k + p]
    yshifted = np.lib.stride_tricks.as_strided(ypad, (R*M, L), 
                                               (ypad.itemsize, ypad.itemsize))
    
    demodpad = pyfftw.n_byte_align(np.zeros((R*M, N), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)
    
    def codefreq2_strided(y):
        ypad[:R*M:R] = y
        np.multiply(yshifted, s_conj, demodpad[:, :L])
        fft.execute() # input is demodpad, output is x_aligned
        x = np.array(x_aligned.T) # we need a copy, which np.array provides
        return x
    
    return codefreq2_strided

def CodeFreq2Cython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    demodpad = pyfftw.n_byte_align(np.zeros((R*M, N), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)
    
    #demodpad = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    #x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    #fft = pyfftw.FFTW(demodpad, x_aligned, axes=(0,), threads=_THREADS)
    
    return libpoint_adjoint.CodeFreq2Cython(s, demodpad, x_aligned, fft, M, R)