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
import scipy.sparse as sparse
import pyfftw
import multiprocessing

from . import libpoint_forward
from .time_varying_conv import tvconv_by_output
from .common import model_dec

_THREADS = multiprocessing.cpu_count()

__all__ = ['DirectSum', 'DirectSumCython', 'CodeFreqSparse',
           'FreqCodeStrided', 'FreqCodeCython', 'FreqCodeNumba']

def forward_factory_dec(fun):
    doc = r"""Construct a point model function.

    This function models a radar scene using a sum of independent point
    scatterers located on a regular grid in delay-frequency space. This
    model is assumed by the typical point target delay-frequency matched
    filter; in fact, they are adjoint operations.

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
        Number of frequency steps, equal to the length of the input's
        first dimension.

    M : int
        Length of the output for the forward operation.

    R : int
        Undersampling ratio, the sampling rate of the transmitted signal
        over the sampling rate of the measured signal.


    Returns
    -------

    forward : function
        Forward operator function that transforms a delay-frequency radar
        scene `x` into a modeled signal `y`.


    See Also
    --------

    Adjoint : Corresponding adjoint model.
    Forward_alt : Slightly different forward model.


    Notes
    -----

    The returned function implements the equation:

    .. math::

        y[m] = \frac{1}{\sqrt{N}} \sum_{n,p} ( e^{2 \pi i n m / N}
                                               s[R m - p + L - 1]
                                               x[n, p] )

    for input `x` and specified `s`, `N`, `R`, and `L` = len(`s`).

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding adjoint operator is well-scaled in the sense that the
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

        doc = r"""Apply point model to input.

        This function models a radar scene using a sum of independent point
        scatterers located on a regular grid in delay-frequency space. This
        model is assumed by the typical point target delay-frequency matched
        filter; in fact, they are adjoint operations.


        Parameters
        ----------

        x : 2-D ndarray with `shape`=`inshape` and `dtype`=`indtype`
            Complex values giving point target reflectivity and phase at each
            location in delay-frequency space. The first axis indexes
            frequency, while the seconds indexes delay.


        Returns
        -------

        y : 1-D ndarray with `shape`=`outshape` and `dtype`=`outdtype`
            Complex values representing a measured radar signal.


        Attributes
        ----------

        s : 1-D ndarray
            Transmitted pulse signal, defining the encoding and length of
            the pulse (by the sample length of the array).

        N : int
            Number of frequency steps, equal to the length of the input's
            first dimension.

        R : int
            Undersampling ratio, the sampling rate of the transmitted signal
            over the sampling rate of the measured signal.

        freqs : 1-D ndarray
            Normalized frequency index for the first axis of the input,
            equivalent to `np.fft.fftfreq(N, d=1.0)`. To find the Doppler
            frequencies, multiply by the sampling frequency (divide by
            sampling period).

        delays : 1-D ndarray
            Delay index for the second axis of the input, giving the number
            of samples by which each filtered sample is delayed relative to
            the beginning of the output.

        inshape, outshape : tuple
            Tuples giving the shape of the input and output arrays,
            respectively.

        indtype, outdtype : dtype
            Dtypes of the input and output arrays, respectively.


        Notes
        -----

        The returned function implements the equation:

        .. math::

            y[m] = \frac{1}{\sqrt{N}} \sum_{n,p} ( e^{2 \pi i n m / N}
                                                   s[R m - p + L - 1]
                                                   x[n, p] )

        for input `x` and specified `s`, `N`, `R`, and `L` = len(`s`).

        The :math:`1/\sqrt{N}` term is included so that composition with the
        corresponding adjoint operator is well-scaled in the sense that the
        central diagonal entries of the composed Forward-Adjoint operation
        matrix are equal to the norm of `s`.

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
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    @forward_op_dec(s, N, M, R)
    def direct_sum(x):
        y = np.zeros(M, dtype=xydtype)
        for m in xrange(M):
            ym = 0
            for n in xrange(N):
                phase_over_sqrtN = (np.exp(2*np.pi*1j*n*m/N)/np.sqrt(N)).astype(xydtype)
                # constraints on p from bounds of s:
                # Rm - p + L - 1 >= 0:
                #       p <= Rm + L - 1 < Rm + L
                # Rm - p + L - 1 <= L - 1:
                #       p >= Rm
                # but constraints on p from bounds of x imply 0 <= p < P = R*M + L - R
                # so pstart = max(0, R*m) = R*m
                #    pstop = min(P, R*m + L) = min(R*M + L - R, R*m + L) = R*m + L
                for p in xrange(R*m, R*m + L):
                    ym += phase_over_sqrtN*s[R*m - p + L - 1]*x[n, p]
            y[m] = ym

        return y

    return direct_sum

@forward_factory_dec
def DirectSumCython(s, N, M, R=1):
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')

    idftmat = np.exp(2*np.pi*1j*np.arange(N)*np.arange(M)[:, None]/N)/np.sqrt(N)
    idftmat = idftmat.astype(xydtype) # use precision of output

    return forward_op_dec(s, N, M, R)(libpoint_forward.DirectSumCython(s, idftmat, R))

# CodeFreq implementations apply the code modulation first, then the
# Fourier frequency synthesis

@forward_factory_dec
def CodeFreqSparse(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    #         |<------------------P----------------->|
    #         |<--(RM-R)-->| |<-----------L--------->|   |<--(RM-R)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    spad = np.hstack((np.zeros(R*M - R, s.dtype), s, np.zeros(R*M - R, s.dtype)))

    # R == 1, smat =
    # [s[L-1] s[L-2] ... s[0]  0   ...   0    ...  0    0
    #    0    s[L-1] ... s[1] s[0] ...   0    ...  0    0
    #    0      0    ... s[2] s[1] ...   0    ...  0    0
    #    :      :         :    :         :         :    :
    #    0      0    ...  0    0   ... s[L-2] ... s[0]  0
    #    0      0    ...  0    0   ... s[L-1] ... s[1] s[0] ]
    #    |<-------(M-1)------->|         |<-----L------>|
    #    |<----------------(M + L - 1)----------------->|

    # R > 1, smat =
    #    |<-------------L------------->|       |<-------------(RM-R)---------------
    # [s[L-1] ... s[L-R] s[L-1-R] ... s[0]     0    ...  0     0    ...  0   0
    #    0    ...   0     s[L-1]  ... s[R]  s[R-1]  ... s[0]   0    ...  0   0
    #    0    ...   0       0     ... s[2R] s[2R-1] ... s[R] s[R-1] ... s[0] 0
    #    :          0       :          :       :         :     :         :   :
    #    0    ...   0       0     ...  0       0    ...  0     0    ...  0   0
    #
    #                  ...  ...     ...        ...     ...       ...   ...       ...
    #                  ...   :       :          :       :         :     :         :
    #                  ... s[L-R] s[L-1-R] ... s[R]  s[R-1]  ... s[0]   0    ...  0
    #                  ...   0     s[L-1]  ... s[2R] s[2R-1] ... s[R] s[R-1] ... s[0] ]
    # --------(RM-R)-------->|        |<--------------------L-------------------->|
    # -------------------------------(RM + L - R)-------------------------------->|

    smat = np.lib.stride_tricks.as_strided(spad[(P - 1):],
                                           (M, P),
                                           (R*spad.itemsize, -spad.itemsize))
    smat = sparse.csr_matrix(smat)

    idftmat = np.exp(2*np.pi*1j*np.arange(N)*np.arange(M)[:, np.newaxis]/N)/np.sqrt(N)
    idftmat = idftmat.astype(xydtype) # use precision of output

    @forward_op_dec(s, N, M, R)
    def codefreq_sparse(x):
        if sparse.issparse(x):
            return np.asarray((smat*x.T).multiply(idftmat).sum(axis=1)).squeeze()
        else:
            return np.asarray(np.multiply(smat*x.T, idftmat).sum(axis=1)).squeeze()

    return codefreq_sparse

# FreqCode implementations apply the Fourier frequency synthesis first by performing
# an IFFT, then proceed with code modulation

@forward_factory_dec
def FreqCodeStrided(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # need to include 1/sqrt(N) factor, and only easy place is in s
    s_over_sqrtN = s/np.sqrt(N)

    #         |<------------------P----------------->|
    #         |<--(RM-R)-->| |<-----------L--------->|   |<--(RM-R)-->|
    # spad = [0 0 0 .... 0 0 s[0] s[1] ... s[L-2] s[L-1] 0 0 .... 0 0 0]
    spad = np.hstack((np.zeros(R*M - R, s.dtype), s_over_sqrtN, np.zeros(R*M - R, s.dtype)))

    # R == 1, smat =
    # [s[L-1] s[L-2] ... s[0]  0   ...   0    ...  0    0
    #    0    s[L-1] ... s[1] s[0] ...   0    ...  0    0
    #    0      0    ... s[2] s[1] ...   0    ...  0    0
    #    :      :         :    :         :         :    :
    #    0      0    ...  0    0   ... s[L-2] ... s[0]  0
    #    0      0    ...  0    0   ... s[L-1] ... s[1] s[0] ]
    #    |<-------(M-1)------->|         |<-----L------>|
    #    |<----------------(M + L - 1)----------------->|

    # R > 1, smat =
    #    |<-------------L------------->|       |<-------------(RM-R)---------------
    # [s[L-1] ... s[L-R] s[L-1-R] ... s[0]     0    ...  0     0    ...  0   0
    #    0    ...   0     s[L-1]  ... s[R]  s[R-1]  ... s[0]   0    ...  0   0
    #    0    ...   0       0     ... s[2R] s[2R-1] ... s[R] s[R-1] ... s[0] 0
    #    :          0       :          :       :         :     :         :   :
    #    0    ...   0       0     ...  0       0    ...  0     0    ...  0   0
    #
    #                  ...  ...     ...        ...     ...       ...   ...       ...
    #                  ...   :       :          :       :         :     :         :
    #                  ... s[L-R] s[L-1-R] ... s[R]  s[R-1]  ... s[0]   0    ...  0
    #                  ...   0     s[L-1]  ... s[2R] s[2R-1] ... s[R] s[R-1] ... s[0] ]
    # --------(RM-R)-------->|        |<--------------------L-------------------->|
    # -------------------------------(RM + L - R)-------------------------------->|

    smat = np.lib.stride_tricks.as_strided(spad[(P - 1):],
                                           (M, P),
                                           (R*spad.itemsize, -spad.itemsize))

    x_aligned = pyfftw.n_byte_align(np.zeros((N, P), xydtype), pyfftw.simd_alignment)
    X = pyfftw.n_byte_align(np.zeros((N, P), xydtype), pyfftw.simd_alignment)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD',
                       axes=(0,), threads=_THREADS)
    # so we don't have to allocate new memory every time we multiply smat with X
    sX = np.zeros((M, P), xydtype)

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
        # allocate memory for storing duplicated IFFT result
        X_dup = np.zeros((M, P), xydtype)

        def freqcode_strided(x):
            x_aligned[:, :] = x
            ifft.execute() # input is x_aligned, output is X
            X.take(ifftidx, axis=0, out=X_dup, mode='wrap')
            np.multiply(smat, X_dup, out=sX)
            y = sX.sum(axis=1)
            return y

    return forward_op_dec(s, N, M, R)(freqcode_strided)

@forward_factory_dec
def FreqCodeNumba(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # need to include 1/sqrt(N) factor, and only easy place is in s
    s_over_sqrtN = s/np.sqrt(N)

    x_aligned = pyfftw.n_byte_align(np.zeros((P, N), xydtype), pyfftw.simd_alignment)
    X = pyfftw.n_byte_align(np.zeros_like(x_aligned), pyfftw.simd_alignment)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD', threads=_THREADS)

    @forward_op_dec(s, N, M, R)
    def freqcode_numba(x):
        x_aligned[:, :] = x.T
        ifft.execute() # input is x_aligned, output is X
        y = tvconv_by_output(s_over_sqrtN, X, R)
        return y

    return freqcode_numba

@forward_factory_dec
def FreqCodeCython(s, N, M, R=1):
    L = len(s)
    P = R*M + L - R
    # use precision (single or double) of s
    # input and output are always complex
    xydtype = np.result_type(s.dtype, np.complex64)

    # ensure that s is C-contiguous as required by the Cython function
    s = np.asarray(s, order='C')

    x_aligned = pyfftw.n_byte_align(np.zeros((N, P), xydtype), pyfftw.simd_alignment)
    X = pyfftw.n_byte_align(np.zeros((N, P), xydtype), pyfftw.simd_alignment)
    ifft = pyfftw.FFTW(x_aligned, X, direction='FFTW_BACKWARD',
                       axes=(0,), threads=_THREADS)

    fun = libpoint_forward.FreqCodeCython(s, x_aligned, X, ifft, M, R)
    return forward_op_dec(s, N, M, R)(fun)
