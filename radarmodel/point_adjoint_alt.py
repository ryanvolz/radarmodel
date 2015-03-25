#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'radarmodel' developers (see AUTHORS file)
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

from . import libpoint_adjoint_alt
from .delay_multiply import delaymult_like_arg2_prealloc
from .common import model_dec

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'CodeFreqStrided', 'CodeFreqCython', 'CodeFreqNumba']

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

    Forward_alt : Corresponding forward model.
    Adjoint : Slightly different adjoint model.


    Notes
    -----

    The returned function implements the equation:

    .. math::

        x[n, p] &= \frac{1}{\sqrt{N}} \sum_m (
                    e^{-2 \pi i n (R m - p + L - 1)/N}
                    s^*[R m - p + L - 1]
                    y[m] ) \\
                &= \frac{1}{\sqrt{N}} \sum_l (
                    e^{-2 \pi i n l / N}
                    s^*[l]
                    y_R[l + p - (L - 1)] )

    where :math:`y_R` is `y` upsampled by `R` (insert `R`-1 zeros after
    each original element) for input `y` and specified `s`, `N`, `R`, and
    `L` = len(`s`).

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

        The returned function implements the equation:

        .. math::

            x[n, p] &= \frac{1}{\sqrt{N}} \sum_m (
                        e^{-2 \pi i n (R m - p + L - 1)/N}
                        s^*[R m - p + L - 1]
                        y[m] ) \\
                    &= \frac{1}{\sqrt{N}} \sum_l (
                        e^{-2 \pi i n l / N}
                        s^*[l]
                        y_R[l + p - (L - 1)] )

        where :math:`y_R` is `y` upsampled by `R` (insert `R`-1 zeros after
        each original element) for input `y` and specified `s`, `N`, `R`, and
        `L` = len(`s`).

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
                    l = R*m - p + L - 1
                    xnp += s_conj[l]*np.exp(-2*np.pi*1j*n*l/N)*y[m]/np.sqrt(N)
                x[n, p] = xnp

        return x

    return direct_sum

# CodeFreq implementations apply the code demodulation first, then the
# Fourier frequency analysis via FFT

@adjoint_factory_dec
def CodeFreqStrided(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step

    # need to include 1/sqrt(N) factor, and only easy place is in s
    s = s/np.sqrt(N)
    s_conj = s.conj()

    # ypad is y upsampled by R, then padded so there are L-1 zeros at both ends:
    #                     |<----------------RM---------------->|
    #         |<-(L-1)->| |<---R---->|            |<-----R---->| |<-(L-R)->|
    # ypad = [0 0 ... 0 0 y[0] 0 ... 0 y[1] 0 ... y[M-1] 0 ... 0 0 0 ... 0 0]
    #                                                    |<-----(L-1)----->|
    #         \<---------------------(RM + 2L - R - 1)-------------------->|
    #                                   (P + L - 1)
    ypad = np.zeros(P + L - 1, xydtype)
    y_R = ypad[(L - 1):(R*M + L - 1)]

    # yshifted[p, k] = y_R[k + p - (L - 1)] = ypad[k + p]
    # R == 1, M > L,
    # yshifted = [    0        0     ...   0     y[0]   ---  ---
    #                 0        0     ...  y[0]   y[1]    |    |
    #                 :        :           :      :      L    |
    #                 0       y[0]   ... y[L-3] y[L-2]   |    |
    #                y[0]     y[1]   ... y[L-2] y[L-1]  ---   |
    #                 :        :           :      :      |  M+L-1
    #               y[M-L]  y[M-L+1] ... y[M-2] y[M-1]   |    |
    #              y[M-L+1] y[M-L+2] ... y[M-1]   0      M    |
    #                 :        :           :      :      |    |
    #               y[M-2]   y[M-1]  ...   0      0      |    |
    #               y[M-1]     0     ...   0      0   ] ---  ---
    #                 |<------------L------------>|
    #
    # for R > 1, imagine above with y replaced by y_R and indexes going to
    # y_R[RM-R] = y[M-1] in bottom left corner, total of P=(R*M + L - R) rows

    yshifted = np.lib.stride_tricks.as_strided(ypad, (P, L),
                                               (ypad.itemsize, ypad.itemsize))

    demodpad = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)

    @adjoint_op_dec(s, N, M, R)
    def codefreq_strided(y):
        y_R[::R] = y
        np.multiply(yshifted, s_conj, demodpad[:, :L])
        fft.execute() # input is demodpad, output is x_aligned
        x = np.array(x_aligned[:, ::step].T) # we need a copy, which np.array provides
        return x

    return codefreq_strided

@adjoint_factory_dec
def CodeFreqNumba(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step

    # need to include 1/sqrt(N) factor, and only easy place is in s
    s_over_sqrtN = s/np.sqrt(N)

    demodpad = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)

    @adjoint_op_dec(s, N, M, R)
    def codefreq_numba(y):
        delaymult_like_arg2_prealloc(y, s_over_sqrtN, R, demodpad[:, :L])
        fft.execute() # input is demodpad, output is x_aligned
        x = np.array(x_aligned[:, ::step].T) # we need a copy, which np.array provides
        return x

    return codefreq_numba

@adjoint_factory_dec
def CodeFreqCython(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step

    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')

    demodpad = pyfftw.n_byte_align(np.zeros((P, nfft), xydtype), 16)
    x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, x_aligned, threads=_THREADS)

    #demodpad = pyfftw.n_byte_align(np.zeros((nfft, P), xydtype), 16)
    #x_aligned = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    #fft = pyfftw.FFTW(demodpad, x_aligned, axes=(0,), threads=_THREADS)

    fun = libpoint_adjoint_alt.CodeFreqCython(s, demodpad, x_aligned, fft, step, N, M, R)
    return adjoint_op_dec(s, N, M, R)(fun)
