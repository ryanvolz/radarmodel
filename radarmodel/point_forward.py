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

from radarmodel import libpoint_forward

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'DirectSumCython',
           'CodeFreqSparse', 'FreqCodeStrided', 'FreqCodeCython']

# These Point forward models implement the equation:
#     y[m] = \sum_{n,p} 1/N * e^{2*\pi*i*n*m/N} * s[R*m - p] * x[n, p]
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
            for n in xrange(N):
                phase_over_N = np.exp(2*np.pi*1j*n*m/N)/N
                for p in xrange(max(0, R*m - L + 1), min(R*M, R*m + 1)):
                    ym += s[R*m - p]*phase_over_N*x[n, p]
            y[m] = ym

        return y
    
    return direct_sum

def DirectSumCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')
    
    idftmat = np.exp(2*np.pi*1j*np.arange(N)*np.arange(M)[:, None]/N)/N
    idftmat = idftmat.astype(xydtype) # use precision of output
    
    return libpoint_forward.DirectSumCython(s, idftmat, R)

# CodeFreq implementations apply the code modulation first, then the 
# Fourier frequency synthesis

def CodeFreqSparse(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    #         |<--(RM-1)-->| |<-----------L----------->| |<--(RM-L)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    #         |<-----RM------>|<-----------------RM------------------>|
    spad = np.hstack((np.zeros(R*M - 1, s.dtype), s, np.zeros(R*M - L, s.dtype)))
    
    # smat = [s[0]    0     ...  0     0    ...  0   0 ... 0   0    ...  0    0   0 ... 0
    #         s[R]  s[R-1]  ... s[0]   0    ...  0   0 ... 0   0    ...  0    0   0 ... 0
    #         s[2R] s[2R-1] ... s[R] s[R-1] ... s[0] 0 ... 0   0    ...  0    0   0 ... 0
    #          :      :          :     :         :   :     :   :         :    :   : ... :
    #          0      0     ...  0     0    ...  0   0 ... 0 s[L-1] ... s[1] s[0] 0 ... 0 ]
    #                                                                             |<--->|
    #                                                                               R-1
    smat = np.lib.stride_tricks.as_strided(spad[(R*M - 1):], (M, R*M),
                                           (R*spad.itemsize, -spad.itemsize))
    smat = sparse.csr_matrix(smat)

    idftmat = np.exp(2*np.pi*1j*np.arange(N)*np.arange(M)[:, np.newaxis]/N)/N
    idftmat = idftmat.astype(xydtype) # use precision of output
    
    def codefreq_sparse(x):
        if sparse.issparse(x):
            return np.asarray((smat*x.T).multiply(idftmat).sum(axis=1)).squeeze()
        else:
            return np.asarray(np.multiply(smat*x.T, idftmat).sum(axis=1)).squeeze()
    
    return codefreq_sparse

# FreqCode implementations apply the Fourier frequency synthesis first by performing
# an IFFT, then proceed with code modulation

def FreqCodeStrided(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    #         |<--(RM-1)-->| |<-----------L----------->| |<--(RM-L)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    #         |<-----RM------>|<-----------------RM------------------>|
    spad = np.hstack((np.zeros(R*M - 1, s.dtype), s, np.zeros(R*M - L, s.dtype)))
    
    # smat = [s[0]    0     ...  0     0    ...  0   0 ... 0   0    ...  0    0   0 ... 0
    #         s[R]  s[R-1]  ... s[0]   0    ...  0   0 ... 0   0    ...  0    0   0 ... 0
    #         s[2R] s[2R-1] ... s[R] s[R-1] ... s[0] 0 ... 0   0    ...  0    0   0 ... 0
    #          :      :          :     :         :   :     :   :         :    :   : ... :
    #          0      0     ...  0     0    ...  0   0 ... 0 s[L-1] ... s[1] s[0] 0 ... 0 ]
    #                                                                             |<--->|
    #                                                                               R-1
    smat = np.lib.stride_tricks.as_strided(spad[(R*M - 1):], (M, R*M),
                                           (R*spad.itemsize, -spad.itemsize))
    # IFFT below from FFTW does not include 1/N factor, so include it in smat
    smat = smat/N
    
    x_aligned = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    X = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD',
                       axes=(0,), threads=_THREADS)
    # so we don't have to allocate new memory every time we multiply smat with X
    sX = np.zeros((M, R*M), xydtype)
    
    if M <= N:
        # can truncate the IFFT result to length M
        
        def freqcode_strided(x):
            x_aligned[:, :] = x
            ifft.execute() # input is x_aligned, output is X
            X_trunc = X[:M, :]
            np.multiply(smat, X_trunc, out=sX)
            y = sX.sum(axis=1)
            return y
    else:
        # have to duplicate IFFT result by wrapping with modulus N up to length M
        ifftidx = np.arange(M)
        X_dup = np.zeros((M, R*M), xydtype) # allocate memory for storing duplicated IFFT result
        
        def freqcode_strided(x):
            x_aligned[:, :] = x
            ifft.execute() # input is x_aligned, output is X
            X.take(ifftidx, axis=0, out=X_dup, mode='wrap')
            np.multiply(smat, X_dup, out=sX)
            y = sX.sum(axis=1)
            return y
    
    return freqcode_strided

def FreqCodeCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')
    
    x_aligned = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    X = pyfftw.n_byte_align(np.zeros((N, R*M), xydtype), 16)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD',
                       axes=(0,), threads=_THREADS)
    
    return libpoint_forward.FreqCodeCython(s, x_aligned, X, ifft, M, R)