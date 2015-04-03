# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import pyfftw
import multiprocessing

from .delay_multiply import (
    delaymult_like_arg1_prealloc,
    delaymult_like_arg2_prealloc,
)
from .time_varying_conv import (
    tvconv_by_input_prealloc,
    tvconv_by_output_prealloc,
)
from .operator_class import LinearOperator

_THREADS = multiprocessing.cpu_count()

__all__ = [
    'PointGrid',
    'RxRef', 'TxRef',
    'rxref_forward', 'rxref_forward_delaytime',
    'rxref_adjoint_x',
    'txref_forward', 'txref_forward_delaytime',
    'txref_adjoint_x',
]


def reflectivity_freq2time(x, ifft):
    xtime = ifft.get_output_array()

    # calculate ifft while coercing x to same shape/alignment as required
    ifft(x, normalise_idft=False) # output is xtime

    return xtime


# ****************************************************************************
# *********************** TX-referenced operators ****************************
# ****************************************************************************
def txref_forward_delaytime(s, x, y):
    N = x.shape[1]

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    tvconv_by_input_prealloc(s_over_sqrtN, x, y)

    return y

def txref_forward(s, x, ifft, y):
    xtime = reflectivity_freq2time(x, ifft)
    y = txref_forward_delaytime(s, xtime, y)

    return y

def txref_adjoint_x(y, s, fft, x):
    delaymult_out = fft.get_input_array()
    N = x.shape[1]

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    L = len(s)
    delaymult_like_arg2_prealloc(y, s_over_sqrtN, delaymult_out[:, :L])

    fft(output_array=x) # input is delaymult_out

    return x


# ****************************************************************************
# *********************** RX-referenced operators ****************************
# ****************************************************************************
def rxref_forward_delaytime(s, x, y):
    N = x.shape[1]

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    tvconv_by_output_prealloc(s_over_sqrtN, x, y)

    return y

def rxref_forward(s, x, ifft, y):
    xtime = reflectivity_freq2time(x, ifft)
    y = pointgrid_rxref_forward_delaytime(s, xtime, y)

    return y

def rxref_adjoint_x(y, s, fft, N, x_up):
    delaymult_out = fft.get_input_array()
    nfft = x_up.shape[1]
    step = nfft//N

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    M = len(y)
    delaymult_like_arg1_prealloc(y, s_over_sqrtN, delaymult_out[:, :M])

    fft(output_array=x_up) # input is delaymult_out

    # subsample to get desired output size
    return x_up[:, ::step]


# ****************************************************************************
# ********************* Model and Operator Base Classes **********************
# ****************************************************************************
class _PointGrid(object):
    """

    """
    def __init__(self, L, M, N, R, sdtype):
        self._L = L
        self._M = M
        self._N = N
        self._R = R
        self._sdtype = sdtype

        self._P = R*M + L - R

        # x and y are always complex, with given precision of s
        self._xydtype = np.result_type(sdtype, np.complex64)

        self._fft = self._create_fft_plan()
        self._ifft = self._create_ifft_plan()

    def _create_fft_plan(self):
        delaymult_out = pyfftw.n_byte_align(
            np.zeros((self._P, self._N), self._xydtype),
            pyfftw.simd_alignment
        )
        fft_out = pyfftw.n_byte_align(
            np.zeros_like(delaymult_out),
            pyfftw.simd_alignment
        )
        fft = pyfftw.FFTW(delaymult_out, fft_out, threads=_THREADS)

        return fft

    def _create_ifft_plan(self):
        x = pyfftw.n_byte_align(
            np.zeros((self._P, self._N), self._xydtype),
            pyfftw.simd_alignment
        )
        ifft_out = pyfftw.n_byte_align(
            np.zeros_like(x),
            pyfftw.simd_alignment
        )
        ifft = pyfftw.FFTW(x, ifft_out, direction='FFTW_BACKWARD',
                           threads=_THREADS)

        return ifft

class _FixedTx(LinearOperator):
    def __init__(self, s, model):
        self.s = s

        self._init_from_model(model)

        inshape = (model._P, model._N)
        indtype = model._xydtype
        outshape = (model._M,)
        outdtype = model._xydtype

        super(_FixedTx, self).__init__(
            inshape, indtype, outshape, outdtype,
        )

    def _init_from_model(self, model):
        self.model = model

        self.R = model._R
        self._fft = model._fft
        self._ifft = model._ifft

        # normalized frequency index for convenience
        self.freqs = np.fft.fftfreq(model._N, d=1.0)

        # filter delay index (delay relative to beginning of s)
        self.delays = np.arange(-(model._L - model._R), model._R*model._M)

class _FixedReflectivity(LinearOperator):
    def __init__(self, x, model):
        self.x = x

        self._init_from_model(model)

        self._xtime = reflectivity_freq2time(x, self._ifft)

        inshape = (model._L,)
        indtype = model._sdtype
        outshape = (model._M,)
        outdtype = model._xydtype

        super(_FixedReflectivity, self).__init__(
            inshape, indtype, outshape, outdtype,
        )

    def _init_from_model(self, model):
        self.model = model

        self.R = model._R
        self._fft = model._fft
        self._ifft = model._ifft



_pointgrid_init_docstring = """

"""

# ****************************************************************************
# *************** TX-referenced Model Class and Operators ********************
# ****************************************************************************
class TxRef(_PointGrid):
    def __init__(self, txlen, rxlen, fftlen, rxundersample=1,
                 precision=np.double):
        __doc__ = _pointgrid_init_docstring

        # if N < L, we would need to take FFT with nfft >= L so we don't lose
        # data, then subsample to get our N points that we desire
        # this is a lot of extra work just to throw a lot of it away,
        # so don't allow this case (see RxRef for how it would work)
        if fftlen < txlen:
            raise ValueError('fftlen < txlen does not result in faster'
                             + ' computation, so it is not allowed. Choose'
                             + ' fftlen >= txlen.')

        super(TxRef, self).__init__(
            txlen, rxlen, fftlen, rxundersample, precision,
        )

    def FixedTx(self, s):
        return TxRefFixedTx(s, self)

    def FixedReflectivity(self, x):
        return TxRefFixedReflectivity(x, self)

    def forward(self, s, x, y=None):
        if y is None:
            y = np.empty(self._M, self._xydtype)
        return txref_forward(s, x, self._ifft, y)

    def adjoint_x(self, y, s, x=None):
        if x is None:
            x = np.empty((self._P, self._N), self._xydtype)
        return txref_adjoint_x(y, s, self._fft, x)

    def adjoint_s(self, y, x, s=None):
        raise NotImplementedError

# indicate that default pointgrid model should be the tx-referenced one
# by assigning it to the generic name
PointGrid = TxRef

class TxRefFixedTx(_FixedTx):
    def _forward(self, x, y):
        return txref_forward(self.s, x, self._ifft, y)

    def _adjoint(self, y, x):
        return txref_adjoint_x(y, self.s, self._fft, x)

class TxRefFixedReflectivity(_FixedReflectivity):
    def _forward(self, s, y):
        return txref_forward_delaytime(s, self._xtime, y)

    def _adjoint(self, y, s):
        raise NotImplementedError


# ****************************************************************************
# *************** RX-referenced Model Class and Operators ********************
# ****************************************************************************
class RxRef(_PointGrid):
    def __init__(self, txlen, rxlen, fftlen, rxundersample=1,
                 precision=np.double):
        __doc__ = _pointgrid_init_docstring

        # when N < M, need to take FFT with nfft >= M so we don't lose data
        # then subsample to get our N points that we desire
        self._step = rxlen // fftlen + 1
        self._nfft = fftlen*self._step

        super(RxRef, self).__init__(
            txlen, rxlen, fftlen, rxundersample, precision,
        )

    def _create_fft_plan(self):
        delaymult_out = pyfftw.n_byte_align(
            np.zeros((self._P, self._nfft), self._xydtype),
            pyfftw.simd_alignment
        )
        fft_out = pyfftw.n_byte_align(
            np.zeros_like(delaymult_out),
            pyfftw.simd_alignment
        )
        fft = pyfftw.FFTW(delaymult_out, fft_out, threads=_THREADS)

        return fft

    def FixedTx(self, s):
        return RxRefFixedTx(s, self)

    def FixedReflectivity(self, x):
        return RxRefFixedReflectivity(x, self)

    def forward(self, s, x, y=None):
        if y is None:
            y = np.empty(self._M, self._xydtype)
        return rxref_forward(s, x, self._ifft, y)

    def adjoint_x(self, y, s, x_up=None):
        if x_up is None:
            x_up = np.empty((self._P, self._nfft), self._xydtype)
        return rxref_adjoint_x(y, s, self._fft, self._N, x_up)

    def adjoint_s(self, y, x, s=None):
        raise NotImplementedError

class RxRefFixedTx(_FixedTx):
    def __init__(self, s, model):
        # specialize init for RxRef because inshape is different
        self.s = s

        self._init_from_model(model)

        inshape = (model._P, model._nfft)
        indtype = model._xydtype
        outshape = (model._M,)
        outdtype = model._xydtype

        super(_FixedTx, self).__init__(
            inshape, indtype, outshape, outdtype,
        )

    def _forward(self, x, y):
        return rxref_forward(self.s, x, self._ifft, y)

    def _adjoint(self, y, x):
        return rxref_adjoint_x(y, self.s, self._fft, self.model._N, x)

class RxRefFixedReflectivity(_FixedReflectivity):
    def _forward(self, s, y):
        return rxref_forward_delaytime(s, self._xtime, y)

    def _adjoint(self, y, s):
        raise NotImplementedError
