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

from radarmodel import libpoint_forward_alt

from .common import model_dec

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'FreqCodeStrided', 'FreqCodeCython']

def forward_factory_dec(fun):
    doc = """Construct a point model function.
    
    This function models a radar scene using a sum of independent point 
    scatterers located on a regular grid in delay-frequency space. This model 
    is assumed by the typical point target delay-frequency matched filter; 
    in fact, they are adjoint operations.
    
    The returned function implements the equation:
        y[m] = \sum_{n,p} ( 1/sqrt(N) * e^{2*\pi*i*n*(R*m - p + L - 1)/N} 
                                      * s[R*m - p + L - 1] * x[n, p] )
             = \sum_{n,l} ( 1/sqrt(N) * e^{2*\pi*i*n*l/N} 
                                      * s[l] * x[n, R*m - l + L - 1] )
    for input 'x' and specified 's', 'N', 'R', and 'L = len(s)'.
    
    The 1/sqrt(N) term is included so that pre-composition with the 
    corresponding adjoint operator is well-scaled in the sense that the 
    central diagonal entries of the composed Forward-Adjoint operation matrix 
    are equal to the norm of s.
    
    The arguments specify the parameters of the model and the size of input 
    that it should accept. The dtype that the constructed function will accept 
    as input is complex with floating point precision equal to that of 's'.
    
    It is necessary to take N >= len(s) == L in order to ensure that the 
    operator has a consistent norm (equal to the norm of s). In addition, 
    computation is faster when N is a power of 2 since it depends on the FFT 
    algorithm.
    
    
    Arguments
    ---------
    
    s: 1-D ndarray
        Transmitted pulse signal, defining the encoding and length of 
        the pulse (by the sample length of the array).
        
    N: int
        Number of frequency steps, equal to the length of the input's 
        first dimension.
    
    M: int
        Length of the output for the forward operation.
    
    R: int
        Undersampling ratio, the sampling rate of the transmitted signal 
        over the sampling rate of the measured signal.
    
    """
    if fun.__doc__ is not None:
        fun.__doc__ += doc
    else:
        fun.__doc__ = doc
    
    return fun

def forward_op_dec(s, N, M, R):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    inshape = (N, P)
    outshape = (M,)
    
    # normalized frequency index for convenience
    nidx = np.fft.fftfreq(N, d=1.0)
    
    # filter delay index (delay relative to beginning of s)
    pidx = np.arange(-(L - R), R*M)
    
    def decor(fun):
        fun = model_dec(inshape, xydtype, outshape, xydtype)(fun)
        
        fun.s = s
        fun.N = N
        fun.R = R
        fun.freqs = nidx
        fun.delays = pidx
        
        doc = """Apply point model to input.
        
        This function models a radar scene using a sum of independent point 
        scatterers located on a regular grid in delay-frequency space. This 
        model is assumed by the typical point target delay-frequency matched 
        filter; in fact, they are adjoint operations.
        
        This function implements the equation:
            y[m] = \sum_{n,p} ( 1/sqrt(N) * e^{2*\pi*i*n*(R*m - p + L - 1)/N} 
                                          * s[R*m - p + L - 1] * x[n, p] )
                 = \sum_{n,l} ( 1/sqrt(N) * e^{2*\pi*i*n*l/N} 
                                          * s[l] * x[n, R*m - l + L - 1] )
        for input 'x' and specified 's', 'N', 'R', and 'L = len(s)'.
        
        The 1/sqrt(N) term is included so that pre-composition with the 
        corresponding adjoint operator is well-scaled in the sense that the 
        central diagonal entries of the composed Forward-Adjoint operation 
        matrix are equal to the norm of s.
        
        
        Input
        -----
        
        x: 2-D ndarray with shape=inshape and dtype=indtype
            Complex values giving point target reflectivity and phase at each 
            location in delay-frequency space. The first axis indexes 
            frequency, while the seconds indexes delay.
        
        Output
        ------
        
        y: 1-D ndarray with shape=outshape and dtype=outdtype
            Complex values representing a measured radar signal.
        
        Attributes
        ----------
        
        s: 1-D ndarray
            Transmitted pulse signal, defining the encoding and length of 
            the pulse (by the sample length of the array).
            
        N: int
            Number of frequency steps, equal to the length of the input's 
            first dimension.
        
        R: int
            Undersampling ratio, the sampling rate of the transmitted signal 
            over the sampling rate of the measured signal.
        
        freqs: 1-D ndarray
            Normalized frequency index for the first axis of the input, 
            equivalent to np.fft.fftfreq(N, d=1.0). To find the Doppler 
            frequencies, multiply by the sampling frequency (divide by 
            sampling period).
        
        delays: 1-D ndarray
            Delay index for the second axis of the input, giving the number 
            of samples by which each filtered sample is delayed relative to 
            the beginning of the output.
        
        inshape/outshape: tuple
            Tuples giving the shape of the input and output arrays, 
            respectively.
        
        indtype/outdtype: dtype
            Dtypes of the input and output arrays, respectively.
        
        """
        
        if fun.__doc__ is not None:
            fun.__doc__ += doc
        else:
            fun.__doc__ = doc
        
        return fun
    return decor

@forward_factory_dec
def DirectSum(s, N, M, R=1):
    L = len(s)
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    @forward_op_dec(s, N, M, R)
    def direct_sum(x):
        y = np.zeros(M, dtype=xydtype)
        for m in xrange(M):
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
                s_l = s[l]
                for n in xrange(N):
                    ym += 1/np.sqrt(N)*np.exp(2*np.pi*1j*n*l/N)*s_l*x[n, R*m - l + L - 1]
            y[m] = ym

        return y
    
    return direct_sum

# FreqCode implementations apply the Fourier frequency synthesis first by performing
# an IFFT, then proceed with code modulation

@forward_factory_dec
def FreqCodeStrided(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # when N < L, we need to wrap result of FFT so that we can get L samples
    # do this by upsampling x before taking FFT, which accomplishes this wrapping
    step = L // N + 1
    nfft = N*step
    
    # need to include 1/sqrt(N) factor, and only easy place is in s
    s_over_sqrtN = s/np.sqrt(N)
    
    # we transpose x and write it upsampled into x_aligned to align it for FFT
    x_aligned = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    X = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD', threads=_THREADS)
    
    # Xstrided[m, l] = X[Rm - l + L - 1, l] where negative indices do not wrap
    #
    # Xstrided = [    X[L-1, 0]        X[L-2, 1]     ...    X[1, L-2]       X[0, L-1]
    #                X[R+L-1, 0]      X[R+L-2, 1]    ...   X[R+1, L-2]      X[R, L-1]
    #                X[2R+L-1, 0]     X[2R+L-2, 1]   ...   X[2R+1, L-2]     X[2R, L-1]
    #                     :                :                    :               :
    #              X[R(M-1)+L-1, 0] X[R(M-1)+L-2, 1] ... X[R(M-1)+1, L-2] X[R(M-1), L-1] ]
    
    Xstrided = np.lib.stride_tricks.as_strided(X[(L-1):, :], (M, L),
                                               (R*X.strides[0], 
                                                X.strides[1] - X.strides[0]))

    # so we don't have to allocate new memory every time we multiply s_over_N with Xstrided
    sX = np.zeros((M, L), xydtype)
    
    @forward_op_dec(s, N, M, R)
    def freqcode_strided(x):
        x_aligned[:, ::step] = x.T # upsample along FFT axis by step, results in wrapped FFT
        ifft.execute() # input is x_aligned, output is X
        np.multiply(s_over_sqrtN, Xstrided, out=sX)
        y = sX.sum(axis=1)
        return y
    
    return freqcode_strided

@forward_factory_dec
def FreqCodeCython(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)
    
    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')
    
    x_aligned = pyfftw.n_byte_align(np.zeros((N, P), xydtype), 16)
    X = pyfftw.n_byte_align(np.zeros((N, P), xydtype), 16)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD', 
                       axes=(0,), threads=_THREADS)
    
    fun = libpoint_forward_alt.FreqCodeCython(s, x_aligned, X, ifft, M, R)
    return forward_op_dec(s, N, M, R)(fun)