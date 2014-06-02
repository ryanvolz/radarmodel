#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import numba
from numba.decorators import jit, autojit
import scipy.sparse as sparse
import pyfftw
import multiprocessing

from radarmodel import libpoint_adjoint

from .common import model_dec

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'DirectSumCython', 'DirectSumNumba',
           'FreqCodeSparse', 'CodeFreqStrided', 'CodeFreqCython']

def adjoint_factory_dec(fun):
    doc = r"""Construct an adjoint point model function.

    This adjoint operation acts as a delay-frequency matched filter
    assuming that the radar operates according to the corresponding
    forward model.

    The arguments specify the parameters of the model and the size of
    input that it should accept. The dtype that the constructed function
    will accept as input is complex with floating point precision equal to
    that of `s`.


    Parameters
    ----------

    s : 1-D ndarray
        Transmitted pulse signal, defining the encoding and length of
        the pulse (by the sample length of the array).

    N : int
        Number of frequency steps, equal to the length of the output's
        first dimension.

    M : int
        Length of the input that the adjoint operation should accept.

    R : int
        Undersampling ratio, the sampling rate of the transmitted signal
        over the sampling rate of the measured signal.


    Returns
    -------

    adjoint : function
        Adjoint operator function that transforms a radar signal `y` into
        delay-frequency components `x`.


    See Also
    --------

    Forward : Corresponding forward model.
    Adjoint_alt : Slightly different adjoint model.


    Notes
    -----

    The returned function implements the equation:

    .. math::

        x[n, p] = \frac{1}{\sqrt{N}} \sum_m ( e^{-2 \pi i n m / N}
                                              s^*[R m - p + L - 1]
                                              y[m] )

    for input `y` and specified `s`, `N`, `R`, and `L` = len(`s`).

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding forward operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= len(`s`) == `L` in order to ensure that
    the operator has a consistent norm (equal to the norm of `s`). In
    addition, computation is faster when `N` is a power of 2 since it
    depends on the FFT algorithm.

    """
    if fun.__doc__ is not None:
        fun.__doc__ += doc
    else:
        fun.__doc__ = doc

    return fun

def adjoint_op_dec(s, N, M, R):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    inshape = (M,)
    outshape = (N, P)

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

        doc = r"""Apply adjoint point model to input.

        This adjoint operation acts as a delay-frequency matched filter
        assuming that the radar operates according to the corresponding
        forward model.


        Parameters
        ----------

        y : 1-D ndarray with `shape`=`inshape` and `dtype`=`indtype`
            Complex values representing a measured radar signal (potentially
            as output by the corresponding forward model).


        Returns
        -------

        x : 2-D ndarray with `shape`=`outshape` and `dtype`=`outdtype`
            Delay-frequency matched-filter output, the result of the adjoint
            operation. The first axis indexes frequency, while the second
            indexes delay.


        Attributes
        ----------

        s : 1-D ndarray
            Transmitted pulse signal, defining the encoding and length of
            the pulse (by the sample length of the array).

        N : int
            Number of frequency steps, equal to the length of the output's
            first dimension.

        R : int
            Undersampling ratio, the sampling rate of the transmitted signal
            over the sampling rate of the measured signal.

        freqs : 1-D ndarray
            Normalized frequency index for the first axis of the output,
            equivalent to `np.fft.fftfreq(N, d=1.0)`. To find the Doppler
            frequencies, multiply by the sampling frequency (divide by
            sampling period).

        delays : 1-D ndarray
            Delay index for the second axis of the output, giving the number
            of samples by which each filtered sample is delayed relative to
            the beginning of the input. For output with the same size and
            delay as the input, index with [`delays` >= 0].

        inshape, outshape : tuple
            Tuples giving the shape of the input and output arrays,
            respectively.

        indtype, outdtype : dtype
            Dtypes of the input and output arrays, respectively.


        Notes
        -----

        This function implements the equation:

        .. math::

            x[n, p] = \frac{1}{\sqrt{N}} \sum_m ( e^{-2 \pi i n m / N}
                                                  s^*[R m - p + L - 1]
                                                  y[m] )

        for input `y` and specified `s`, `N`, `R`, and `L` = len(`s`).

        The :math:`1/\sqrt{N}` term is included so that composition with the
        corresponding forward operator is well-scaled in the sense that the
        central diagonal entries of the composed Forward-Adjoint operation
        matrix are equal to the norm of `s`.

        """

        if fun.__doc__ is not None:
            fun.__doc__ += doc
        else:
            fun.__doc__ = doc

        return fun
    return decor

@adjoint_factory_dec
def DirectSum(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    s_conj = s.conj()
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    @adjoint_op_dec(s, N, M, R)
    def direct_sum(y):
        x = np.zeros((N, P), dtype=xydtype)
        for n in xrange(N):
            for p in xrange(P):
                xnp = 0
                # constraints on m from bounds of s:
                # Rm - p + L - 1 >= 0:
                #       m >= ceil((p - L + 1)/R) --> m >= floor((p - L)/R) + 1
                # Rm - p + L - 1 <= L - 1:
                #       m <= floor(p/R)
                for m in xrange(max(0, (p - L)//R + 1), min(M, p//R + 1)):
                    # downshift modulate signal by frequency given by p
                    # then correlate with conjugate of transmitted signal
                    k = R*m - p + L - 1
                    xnp += s_conj[k]*np.exp(-2*np.pi*1j*n*m/N)*y[m]/np.sqrt(N)
                x[n, p] = xnp

        return x

    return direct_sum

@adjoint_factory_dec
def DirectSumNumba(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    s_conj = s.conj()
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    stype = numba.__getattribute__(str(s.dtype))
    xytype = numba.__getattribute__(str(xydtype))

    dftmat = np.exp(-2*np.pi*1j*np.arange(M)*np.arange(N)[:, None]/N)/np.sqrt(N)
    dftmat = dftmat.astype(xydtype) # use precision of output

    @jit(argtypes=[xytype[::1]],
         locals=dict(xnp=xytype))
    def direct_sum_numba(y):
        x = np.zeros((N, P), dtype=y.dtype)
        for n in range(N):
            for p in range(P):
                xnp = 0
                # constraints on m from bounds of s:
                # Rm - p + L - 1 >= 0:
                #       m >= ceil((p - L + 1)/R) --> m >= floor((p - L)/R) + 1
                # Rm - p + L - 1 <= L - 1:
                #       m <= floor(p/R)
                # add R before division so calculation of (p - L)//R + 1 <= 0
                # when it should be with cdivision semantics (floor toward 0)
                mstart = max(0, (p - L + R)//R)
                mstop = min(M, p//R + 1)
                for m in range(mstart, mstop):
                    # downshift modulate signal by frequency given by p
                    # then correlate with conjugate of transmitted signal
                    xnp += s_conj[R*m - p + L - 1]*dftmat[n, m]*y[m]
                x[n, p] = xnp
        return x

    return adjoint_op_dec(s, N, M, R)(direct_sum_numba)

@adjoint_factory_dec
def DirectSumCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')

    dftmat = np.exp(-2*np.pi*1j*np.arange(M)*np.arange(N)[:, None]/N)/np.sqrt(N)
    dftmat = dftmat.astype(xydtype) # use precision of output

    return adjoint_op_dec(s, N, M, R)(libpoint_adjoint.DirectSumCython(s, dftmat, R))

# FreqCode implementations apply the Fourier frequency analysis first by performing
# a frequency downshift, then proceed with code correlation

@adjoint_factory_dec
def FreqCodeSparse(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    #         |<------------------P----------------->|
    #         |<--(RM-R)-->| |<-----------L--------->|   |<--(RM-R)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    spad = np.hstack((np.zeros(R*M - R, s.dtype), s, np.zeros(R*M - R, s.dtype)))

    # R == 1, smatconj =
    # [s*[L-1] s*[L-2] ... s*[0]   0   ...    0    ...   0     0
    #    0     s*[L-1] ... s*[1] s*[0] ...    0    ...   0     0
    #    0        0    ... s*[2] s*[1] ...    0    ...   0     0
    #    :        :          :     :          :          :     :
    #    0        0    ...   0     0   ... s*[L-2] ... s*[0]   0
    #    0        0    ...   0     0   ... s*[L-1] ... s*[1] s*[0] ]
    #    |<------------(M-1)------------>|    |<------L------->|
    #    |<--------------------(M + L - 1)-------------------->|

    # R > 1, smatconj =
    #    |<---------------L--------------->|       |<-------------(RM-R)---------------
    # [s*[L-1] ... s*[L-R] s*[L-1-R] ... s*[0]     0     ...   0      0    ...   0   0
    #    0     ...    0     s*[L-1]  ... s*[R]  s*[R-1]  ... s*[0]    0    ...   0   0
    #    0     ...    0        0     ... s*[2R] s*[2R-1] ... s*[R] s*[R-1] ... s*[0] 0
    #    :            0        :           :       :           :      :          :   :
    #    0     ...    0        0     ...   0       0     ...   0      0    ...   0   0
    #
    #            ...   ...      ...         ...     ...         ...    ...        ...
    #            ...    :        :           :       :           :      :          :
    #            ... s*[L-R] s*[L-1-R] ... s*[R]  s*[R-1]  ... s*[0]    0    ...   0
    #            ...    0     s*[L-1]  ... s*[2R] s*[2R-1] ... s*[R] s*[R-1] ... s*[0] ]
    # --------(RM-R)--->|        |<-----------------------L----------------------->|
    # -------------------------------(RM + L - R)--------------------------------->|

    smatconj = np.lib.stride_tricks.as_strided(spad.conj()[(P - 1):],
                                               (M, P),
                                               (R*spad.itemsize, -spad.itemsize))
    smatconj = sparse.csr_matrix(smatconj)

    dftmat = np.exp(-2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(M)/N)/np.sqrt(N)
    dftmat = dftmat.astype(xydtype) # use precision of output

    @adjoint_op_dec(s, N, M, R)
    def freqcode_sparse(y):
        return np.asarray((dftmat*y)*smatconj)

    return freqcode_sparse

# CodeFreq implementations apply the code demodulation first, then the
# Fourier frequency analysis via FFT

@adjoint_factory_dec
def CodeFreqStrided(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # when N < M, still need to take FFT with nfft >= M so we don't lose data
    # then subsample to get our N points that we desire
    step = M // N + 1
    nfft = N*step

    # need to include 1/sqrt(N) factor, and only easy place is in s
    s = s/np.sqrt(N)

    #         |<------------------P----------------->|
    #         |<--(RM-R)-->| |<----------L---------->|   |<--(RM-R)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    spad = np.hstack((np.zeros(R*M - R, s.dtype), s, np.zeros(R*M - R, s.dtype)))

    # R == 1,
    # smatstar = [ s*[L-1]    0       0    ...    0       0     ---
    #         |    s*[L-2] s*[L-1]    0    ...    0       0      |
    #         L    s*[L-3] s*[L-2] s*[L-1] ...    0       0      |
    #         |       :       :       :           :       :      |
    #        ---    s*[0]   s*[1]   s*[2]  ...    0       0     M-1
    #        ---      0     s*[0]   s*[1]  ...    0       0      |
    #         |       0       0     s*[0]  ...    0       0      |
    #         |       :       :       :           :       :      |
    #         |       0       0       0    ... s*[L-1]    0     ---
    #        M-1      0       0       0    ... s*[L-2] s*[L-1]  ---
    #         |       0       0       0    ... s*[L-3] s*[L-2]   |
    #         |       :       :       :           :       :      L
    #         |       0       0       0    ...  s*[0]   s*[1]    |
    #        ---      0       0       0    ...    0     s*[0] ] ---
    #                 |<----------------M---------------->|

    # R > 1,
    # smatstar = [ s*[L-1]     0       0     ...      0        0     ---
    #         |       :        :       :              :        :      |
    #         |    s*[L-R]     0       0     ...      0        0      |
    #         L   s*[L-1-R] s*[L-1]    0     ...      0        0      |
    #         |       :        :       :              :        :      |
    #         |       :        :       :              :        :      |
    #        ---    s*[0]    s*[R]   s*[2R]  ...      0        0      |
    #        ---      0     s*[R-1] s*[2R-1] ...      0        0      |
    #         |       :        :       :              :        :    RM-R
    #         |       0      s*[0]   s*[R]   ...      0        0      |
    #         |       0        0    s*[R-1]  ...      0        0      |
    #         |       :        :       :              :        :      |
    #         |       0        0     s*[0]   ...      0        0      |
    #         |       0        0       0     ...      0        0      |
    #         |       :        :       :              :        :      |
    #         |       :        :       :              :        :      |
    #       RM-R      0        0       0     ...   s*[L-R]     0     ---
    #         |       0        0       0     ...  s*[L-1-R] s*[L-1]  ---
    #         |       :        :       :              :        :      |
    #         |       :        :       :              :        :      |
    #         |       0        0       0     ...    s*[R]    s*[2R]   |
    #         |       0        0       0     ...   s*[R-1]  s*[2R-1]  |
    #         |       :        :       :              :        :      L
    #         |       0        0       0     ...    s*[0]    s*[R]    |
    #         |       0        0       0     ...      0     s*[R-1]   |
    #         |       :        :       :              :        :      |
    #        ---      0        0       0     ...      0      s*[0] ] ---
    #                 |<-------------------M------------------>|

    smatstar = np.lib.stride_tricks.as_strided(spad.conj()[(P - 1):],
                                               (P, M),
                                               (-spad.itemsize, R*spad.itemsize))

    demodpad = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)

    @adjoint_op_dec(s, N, M, R)
    def codefreq_strided(y):
        np.multiply(smatstar, y, demodpad[:, :M])
        fft.execute() # input is demodpad, output is x_aligned
        x = np.array(x_aligned[:, ::step].T) # need a copy, which np.array provides
        return x

    return codefreq_strided

@adjoint_factory_dec
def CodeFreqCython(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # when N < M, still need to take FFT with nfft >= M so we don't lose data
    # then subsample to get our N points that we desire
    step = M // N + 1
    nfft = N*step

    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')

    demodpad = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)

    fun = libpoint_adjoint.CodeFreqCython(s, demodpad, x_aligned, fft, step, N, M, R)
    return adjoint_op_dec(s, N, M, R)(fun)
