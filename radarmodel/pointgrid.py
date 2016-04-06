# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Radar models based on a delay-frequency grid of point targets.


.. currentmodule:: radarmodel.pointgrid

TX-referenced Model
-------------------

.. autosummary::
    :toctree:

    txref_forward
    txref_forward_delaytime
    txref_adjoint_x
    TxRef


RX-referenced Model
-------------------

.. autosummary::
    :toctree:

    rxref_forward
    rxref_forward_delaytime
    rxref_adjoint_x
    RxRef


References
----------

.. [1]  Volz, Ryan, (2014), "Theory and Applications of Sparsity for Radar
        Sensing of Ionospheric Plasma." Ph.D. Thesis, Stanford University.
        https://github.com/ryanvolz/thesis.


Examples
--------

>>> import numpy as np
>>> from radarmodel import pointgrid
>>> model = pointgrid.TxRef(3, 10, 16, precision=np.float32)
>>> s = np.asarray([1, -1, 1], dtype=model.sdtype)
>>> x = np.zeros((model.P, model.N), model.xydtype)
>>> x[np.nonzero(model.delays == 2), 0] = 1
>>> model.forward(s, x)
array([ 0.00+0.j,  0.00+0.j,  0.25+0.j, -0.25+0.j,  0.25+0.j,  0.00+0.j,
        0.00+0.j,  0.00+0.j,  0.00+0.j,  0.00+0.j], dtype=complex64)
>>> op = model(s=s)
>>> op.forward(x)
array([ 0.00+0.j,  0.00+0.j,  0.25+0.j, -0.25+0.j,  0.25+0.j,  0.00+0.j,
        0.00+0.j,  0.00+0.j,  0.00+0.j,  0.00+0.j], dtype=complex64)
>>> x[np.nonzero(model.delays == 7), 1] = 4
>>> op.forward(x)
array([ 0.00000000+0.j        ,  0.00000000+0.j        ,
        0.25000000+0.j        , -0.25000000+0.j        ,
        0.25000000+0.j        ,  0.00000000+0.j        ,
        0.00000000+0.j        ,  1.00000000+0.j        ,
       -0.92387950-0.38268343j,  0.70710677+0.70710677j], dtype=complex64)
>>> y = op.forward(x)
>>> z = op.adjoint(y)
>>> z[:, 0]
array([ 0.00000000+0.j        ,  0.00000000+0.j        ,
        0.06250000+0.j        , -0.12500000+0.j        ,
        0.18750000+0.j        , -0.12500000+0.j        ,
        0.06250000+0.j        ,  0.25000000+0.j        ,
       -0.48096988-0.09567086j,  0.65774655+0.27244756j,
       -0.40774655-0.27244756j,  0.17677669+0.17677669j], dtype=complex64)

"""

import numpy as np
import pyfftw

from rkl.delay_multiply import (
    delaymult_like_arg1_prealloc,
    delaymult_like_arg2_prealloc,
)
from rkl.time_varying_conv import (
    tvconv_by_input_prealloc,
    tvconv_by_output_prealloc,
)
from .operator_class import LinearOperator

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
    r"""Calculate partial forward operation of the TX-referenced point model.

    This function models a radar scene using a sum of independent point
    scatterers located on a regular grid in delay-time space. This model is
    equivalent to the delay-frequency model of `txref_forward`; the
    reflectivities `x` are just the inverse DFT of the delay-frequency
    reflectivities used in `txref_forward`. A delay-frequency representation
    is often preferred because it is naturally sparse, but this function is
    still used to calculate the delay-frequency forward model after applying
    the inverse DFT to the reflectivities.

    This function implements the equation:

    .. math::

        y[m] &= \frac{1}{\sqrt{N}} \sum_p ( s[R m - p + L - 1]
                                            x[p, R m - p + L - 1] ) \\
             &= \frac{1}{\sqrt{N}} \sum_l ( s[l]
                                            x[R m - l + L - 1, n] )

    for inputs `s` and `x`, and constants `L`, `N`, and `R`.


    Parameters
    ----------

    s : 1-D ndarray, length `L`
        Real or complex values giving the transmitted radar signal.

    x : 2-D ndarray, shape (`P`, `N`)
        Complex values giving the reflectivity of the radar target scene
        as a function of delay (first index) and frequency (second index).

    y : 1-D ndarray, length `M` and dtype the same as `x`
        Array for storing the forward model's output, the received radar
        signal. The length `M` must satisfy :math:`P = RM + L - R` for the
        input lengths `L` and `P` and an integer undersampling ratio `R` which
        is inferred from that equality.


    Returns
    -------

    y : 1-D ndarray, length `M` and dtype the same as `x`
        The received radar signal produced by the transmitted signal `s` and
        the target scene reflectivities `x`. Relative to the transmitted
        signal `s`, the output begins at a time index corresponding to the
        time at which the final component of `s` is transmitted.

    See Also
    --------

    txref_forward : Delay-frequency forward model, which uses this operation.
    .tvconv_by_input : Component of this calculation.


    Notes
    -----

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding adjoint operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= `L` in order to ensure that the operator
    has a consistent norm (equal to the norm of `s`). In addition, computation
    is faster when `N` is a power of 2 since it depends on the FFT algorithm.

    """
    N = x.shape[1]

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    tvconv_by_input_prealloc(s_over_sqrtN, x, y)

    return y

def txref_forward(s, x, ifft, y):
    r"""Calculate forward operation of the TX-referenced point model.

    This function models a radar scene using a sum of independent point
    scatterers located on a regular grid in delay-frequency space. This
    model is assumed by the typical point target delay-frequency matched
    filter; in fact, they are adjoint operations.

    This function implements the equation:

    .. math::

        y[m] &= \frac{1}{\sqrt{N}} \sum_{p,n} (
                    e^{2 \pi i n (R m - p + L - 1)/N}
                    s[R m - p + L - 1]
                    x[p, n]
                 ) \\
             &= \frac{1}{\sqrt{N}} \sum_{l,n} (
                    e^{2 \pi i n l / N}
                    s[l]
                    x[R m - l + L - 1, n]
                 )

    for inputs `s` and `x`, and constants `L`, `N`, and `R`.


    Parameters
    ----------

    s : 1-D ndarray, length `L`
        Real or complex values giving the transmitted radar signal.

    x : 2-D ndarray, shape (`P`, `N`)
        Complex values giving the reflectivity of the radar target scene
        as a function of delay (first index) and frequency (second index).

    ifft : pyfftw.FFTW object
        Pre-planned FFTW object for calculating `P` inverse FFTs of length
        `N` for input of shape (`P`, `N`) and dtype the same as `x`.

    y : 1-D ndarray, length `M` and dtype the same as `x`
        Array for storing the forward model's output, the received radar
        signal. The length `M` must satisfy :math:`P = RM + L - R` for the
        input lengths `L` and `P` and an integer undersampling ratio `R` which
        is inferred from that equality.


    Returns
    -------

    y : 1-D ndarray, length `M` and dtype the same as `x`
        The received radar signal produced by the transmitted signal `s` and
        the target scene reflectivities `x`. Relative to the transmitted
        signal `s`, the output begins at a time index corresponding to the
        time at which the final component of `s` is transmitted.

    See Also
    --------

    txref_adjoint_x : Corresponding adjoint operator w.r.t. reflectivity.
    TxRef : Class for computing the RX-referenced point target grid model.
    txref_forward_delaytime : Delay-time formulation of the forward model.
    .tvconv_by_input : Component of this calculation.
    rxref_forward : Similar forward operator, for the RX-referenced model.


    Notes
    -----

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding adjoint operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= `L` in order to ensure that the operator
    has a consistent norm (equal to the norm of `s`). In addition, computation
    is faster when `N` is a power of 2 since it depends on the FFT algorithm.

    """
    xtime = reflectivity_freq2time(x, ifft)
    y = txref_forward_delaytime(s, xtime, y)

    return y

def txref_adjoint_x(y, s, fft, x):
    r"""Calculate adjoint w.r.t reflectivity of TX-referenced point model.

    This adjoint operation acts as a delay-frequency matched filter
    assuming that the radar operates according to the corresponding
    forward model.

    This function implements the equation:

    .. math::

        x[p, n] &= \frac{1}{\sqrt{N}} \sum_m (
                        e^{-2 \pi i n (R m - p + L - 1)/N}
                        s^*[R m - p + L - 1]
                        y[m]
                    ) \\
                &= \frac{1}{\sqrt{N}} \sum_l (
                        e^{-2 \pi i n l / N}
                        s^*[l]
                        y_R[l + p - (L - 1)]
                    )

    where :math:`y_R` is `y` upsampled by `R` (insert `R`-1 zeros after each
    original element) for inputs `y` and `s`, and constants `L`, `N`, and `R`.


    Parameters
    ----------

    y : 1-D ndarray, length `M`
        Complex values representing a measured radar signal (potentially
        as output by the corresponding forward model).

    s : 1-D ndarray, length `L`
        Real or complex values giving the transmitted radar signal.

    fft : pyfftw.FFTW object
        Pre-planned FFTW object for calculating `P` forward FFTs of length
        `N` for input of shape (`P`, `N`) and dtype the same as `y`.

    x : 2-D ndarray, shape (`P`, `N`) and dtype the same as `y`
        Array for storing the adjoint's output reflectivity. `P` must equal
        :math:`P = RM + L - R` for the input lengths `M` and `L` and an
        integer undersampling ratio `R` which is inferred from that equality.


    Returns
    -------

    x : 2-D ndarray, shape (`P`, `N`) and dtype the same as `y`
        Delay-frequency matched-filter output, the result of the adjoint
        operation. The first axis indexes delay, while the second
        indexes frequency.


    See Also
    --------

    txref_forward : Corresponding forward operator.
    TxRef : Class for computing the TX-referenced point target grid model.
    .delaymult_like_arg2 : Component of this calculation.
    rxref_adjoint_x : Similar adjoint operator, for the RX-referenced model.


    Notes
    -----

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding forward operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= `L` in order to ensure that the operator
    has a consistent norm (equal to the norm of `s`). In addition, computation
    is faster when `N` is a power of 2 since it depends on the FFT algorithm.

    """
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
    r"""Calculate partial forward operation of the RX-referenced point model.

    This function models a radar scene using a sum of independent point
    scatterers located on a regular grid in delay-time space. This model is
    equivalent to the delay-frequency model of `rxref_forward`; the
    reflectivities `x` are just the inverse DFT of the delay-frequency
    reflectivities used in `rxref_forward`. A delay-frequency representation
    is often preferred because it is naturally sparse, but this function is
    still used to calculate the delay-frequency forward model after applying
    the inverse DFT to the reflectivities.

    This function implements the equation:

    .. math::

        y[m] = \frac{1}{\sqrt{N}} \sum_p ( s[R m - p + L - 1]
                                           x[p, R m - p + L - 1] )

    for inputs `s` and `x`, and constants `L`, `N`, and `R`.


    Parameters
    ----------

    s : 1-D ndarray, length `L`
        Real or complex values giving the transmitted radar signal.

    x : 2-D ndarray, shape (`P`, `N`)
        Complex values giving the reflectivity of the radar target scene
        as a function of delay (first index) and frequency (second index).

    y : 1-D ndarray, length `M` and dtype the same as `x`
        Array for storing the forward model's output, the received radar
        signal. The length `M` must satisfy :math:`P = RM + L - R` for the
        input lengths `L` and `P` and an integer undersampling ratio `R` which
        is inferred from that equality.


    Returns
    -------

    y : 1-D ndarray, length `M` and dtype the same as `x`
        The received radar signal produced by the transmitted signal `s` and
        the target scene reflectivities `x`. Relative to the transmitted
        signal `s`, the output begins at a time index corresponding to the
        time at which the final component of `s` is transmitted.

    See Also
    --------

    rxref_forward : Delay-frequency forward model, which uses this operation.
    .tvconv_by_output : Component of this calculation.


    Notes
    -----

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding adjoint operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= `L` in order to ensure that the operator
    has a consistent norm (equal to the norm of `s`). In addition, computation
    is faster when `N` is a power of 2 since it depends on the FFT algorithm.

    """
    N = x.shape[1]

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    tvconv_by_output_prealloc(s_over_sqrtN, x, y)

    return y

def rxref_forward(s, x, ifft, y):
    r"""Calculate forward operation of the RX-referenced point model.

    This function models a radar scene using a sum of independent point
    scatterers located on a regular grid in delay-frequency space. This
    model is assumed by the typical point target delay-frequency matched
    filter; in fact, they are adjoint operations.

    This function implements the equation:

    .. math::

        y[m] = \frac{1}{\sqrt{N}} \sum_{p,n} ( e^{2 \pi i n m / N}
                                               s[R m - p + L - 1]
                                               x[p, n] )

    for inputs `s` and `x`, and constants `L`, `N`, and `R`.


    Parameters
    ----------

    s : 1-D ndarray, length `L`
        Real or complex values giving the transmitted radar signal.

    x : 2-D ndarray, shape (`P`, `N`)
        Complex values giving the reflectivity of the radar target scene
        as a function of delay (first index) and frequency (second index).

    ifft : pyfftw.FFTW object
        Pre-planned FFTW object for calculating `P` inverse FFTs of length
        `N` for input of shape (`P`, `N`) and dtype the same as `x`.

    y : 1-D ndarray, length `M` and dtype the same as `x`
        Array for storing the forward model's output, the received radar
        signal. The length `M` must satisfy :math:`P = RM + L - R` for the
        input lengths `L` and `P` and an integer undersampling ratio `R` which
        is inferred from that equality.


    Returns
    -------

    y : 1-D ndarray, length `M` and dtype the same as `x`
        The received radar signal produced by the transmitted signal `s` and
        the target scene reflectivities `x`. Relative to the transmitted
        signal `s`, the output begins at a time index corresponding to the
        time at which the final component of `s` is transmitted.

    See Also
    --------

    rxref_adjoint_x : Corresponding adjoint operator w.r.t. reflectivity.
    RxRef : Class for computing the RX-referenced point target grid model.
    rxref_forward_delaytime : Delay-time formulation of the forward model.
    .tvconv_by_output : Component of this calculation.
    txref_forward : Similar forward operator, for the TX-referenced model.


    Notes
    -----

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding adjoint operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= `L` in order to ensure that the operator
    has a consistent norm (equal to the norm of `s`). In addition, computation
    is faster when `N` is a power of 2 since it depends on the FFT algorithm.

    """
    xtime = reflectivity_freq2time(x, ifft)
    y = rxref_forward_delaytime(s, xtime, y)

    return y

def rxref_adjoint_x(y, s, fft, x):
    r"""Calculate adjoint w.r.t reflectivity of RX-referenced point model.

    This adjoint operation acts as a delay-frequency matched filter
    assuming that the radar operates according to the corresponding
    forward model.

    This function implements the equation:

    .. math::

        x[p, n] = \frac{1}{\sqrt{N}} \sum_m ( e^{-2 \pi i n m / N}
                                              s^*[R m - p + L - 1]
                                              y[m] )

    for inputs `y` and `s`, and constants `L`, `N`, and `R`.


    Parameters
    ----------

    y : 1-D ndarray, length `M`
        Complex values representing a measured radar signal (potentially
        as output by the corresponding forward model).

    s : 1-D ndarray, length `L`
        Real or complex values giving the transmitted radar signal.

    fft : pyfftw.FFTW object
        Pre-planned FFTW object for calculating `P` forward FFTs of length
        `nfft` for input of shape (`P`, `nfft`) and dtype the same as `y`.
        When `N` is less than `M`, an FFT with a larger length must be used in
        the calculation, so `nfft` is chosen as the smallest integer multiple
        of `N` that is greater than `M`. This integer multiple is the
        frequency subsampling stepsize `step`.

    x : 2-D ndarray, shape (`P`, `N`) and dtype the same as `y`
        Array for storing the adjoint's output reflectivity. `P` must equal
        :math:`P = RM + L - R` for the input lengths `M` and `L` and an
        integer undersampling ratio `R` which is inferred from that equality.


    Returns
    -------

    x : 2-D ndarray, shape (`P`, `N`) and dtype the same as `y`
        Delay-frequency matched-filter output, the result of the adjoint
        operation. The first axis indexes delay, while the second
        indexes frequency.

    See Also
    --------

    rxref_forward : Corresponding forward operator.
    RxRef : Class for computing the RX-referenced point target grid model.
    .delaymult_like_arg1 : Component of this calculation.
    txref_adjoint_x : Similar adjoint operator, for the TX-referenced model.


    Notes
    -----

    The :math:`1/\sqrt{N}` term is included so that composition with the
    corresponding forward operator is well-scaled in the sense that the
    central diagonal entries of the composed Forward-Adjoint operation
    matrix are equal to the norm of `s`.

    It is necessary to take `N` >= `L` in order to ensure that the operator
    has a consistent norm (equal to the norm of `s`). In addition, computation
    is faster when `nfft` is a power of 2 since it depends on the FFT
    algorithm.

    """
    delaymult_out = fft.get_input_array()
    x_up = fft.get_output_array()
    nfft = x_up.shape[1]
    N = x.shape[1]
    step = nfft//N

    # need to include 1/sqrt(N) factor for balanced forward/adjoint
    # and it is most efficient just to apply it to s
    s_over_sqrtN = s/np.sqrt(N)

    M = len(y)
    delaymult_like_arg1_prealloc(y, s_over_sqrtN, delaymult_out[:, :M])

    if step == 1:
        # x_up and x are the same size, save a copy by writing directly to x
        fft(output_array=x) # input is delaymult_out
    else:
        fft() # input is delaymult_out, output is x_up
        # subsample to get desired output size, and copy into x
        x[...] = x_up[:, ::step]

    return x


# ****************************************************************************
# ********************* Model and Operator Base Classes **********************
# ****************************************************************************
class PointGridBase(object):
    """Point grid radar model for specified signal dimensions and types.

    The forward and adjoint operations of the model are evaluated by calling
    the `forward`, `adjoint_x`, and `adjoint_s` methods.

    The function of this class is to increase the speed of repeated
    calculations of the model operations by specifying the problem dimensions
    and precision beforehand. Doing this makes it possible to reserve memory
    for intermediate calculations and create plans for the required FFTs.

    .. automethod:: __init__


    Attributes
    ----------

    delays : 1-D ndarray
        Delay index for the first axis of the target scene reflectivities
        (input to the forward operation and output of the adjoint operation).
        This gives the number of samples by which each reflectivity sample is
        delayed relative to the beginning of the transmitted signal. To get
        the reflectivities with the same delay as the input, index `x` with
        ``[delays >= 0]``.

    freqs : 1-D ndarray
        Normalized frequency index for the second axis of the target scene
        reflectivities (input to the forward operation and output of the
        adjoint operation). This is equivalent to ``np.fft.fftfreq(N, d=1.0)``
        where `N` is the length of the second axis of `x`. To find the Doppler
        frequencies, multiply by the sampling frequency (divide by
        sampling period).

    L : int
        Length of the transmitted signal `s`.

    M : int
        Length of the received radar signal `y`.

    N : int
        Number of frequency steps used to specify the target scene
        reflectivities `x`.

    P : int
        Number of delays used to specify the target scene reflectivities `x`.
        :math:`P = RM + L - R`

    R : int
        Undersampling ratio, the sampling rate of the transmitted signal `s`
        over the sampling rate of the received signal `y`.

    sdtype : dtype
        Array dtype of the transmitted signal `s`.

    sshape : tuple
        Array shape of the transmitted signal `s`, ``(L,)``.

    xshape : tuple
        Array shape of the target reflectivities `x`, ``(P, N)``.

    xydtype : dtype
        Array dtype of the target reflectivities `x` and received signal `y`.
        This dtype is complex but with the same precision as `sdtype`.

    yshape : tuple
        Array shape of the received signal `y`, ``(M,)``.

    """
    def __init__(self, L, M, N, R, precision):
        """Create point grid radar model for specified signal parameters.


        Parameters
        ----------

        L : int
            Length of the transmitted signal `s`.

        M : int
            Length of the received radar signal `y`.

        N : int
            Number of frequency steps used to specify the target scene
            reflectivities `x`.

        R : int
            Undersampling ratio, the sampling rate of the transmitted signal
            `s` over the sampling rate of the received signal `y`.

        precision : dtype
            Dtype of the transmitted signal `s`. The dtypes of the received
            signal `y` and reflectivities `x` are taken to be complex but with
            the same precision.

        """
        self.L = L
        self.M = M
        self.N = N
        self.R = R
        self.sdtype = precision

        self.P = R*M + L - R

        self.sshape = (L,)
        self.xshape = (self.P, N)
        self.yshape = (M,)

        # x and y are always complex, with given precision of s
        self.xydtype = np.result_type(precision, np.complex64)

        self._fft = self._create_fft_plan()
        self._ifft = self._create_ifft_plan()

        # normalized frequency index for convenience
        self.freqs = np.fft.fftfreq(N, d=1.0)

        # filter delay index (delay relative to beginning of s)
        self.delays = np.arange(-(L - R), R*M)

    def _create_fft_plan(self):
        delaymult_out = pyfftw.n_byte_align(
            np.zeros((self.P, self.N), self.xydtype),
            pyfftw.simd_alignment
        )
        fft_out = pyfftw.n_byte_align(
            np.zeros_like(delaymult_out),
            pyfftw.simd_alignment
        )
        fft = pyfftw.FFTW(delaymult_out, fft_out)

        return fft

    def _create_ifft_plan(self):
        x = pyfftw.n_byte_align(
            np.zeros((self.P, self.N), self.xydtype),
            pyfftw.simd_alignment
        )
        ifft_out = pyfftw.n_byte_align(
            np.zeros_like(x),
            pyfftw.simd_alignment
        )
        ifft = pyfftw.FFTW(x, ifft_out, direction='FFTW_BACKWARD')

        return ifft

    def __call__(self, s=None, x=None):
        """Create operator for model with either fixed `s` or `x`.

        Specify either `s` or `x`, but not both.


        Parameters
        ----------

        s : 1-D ndarray
            Real or complex values giving the transmitted radar signal.

        x : 2-D ndarray
            Complex values giving the reflectivity of the radar target scene
            as a function of delay (first index) and frequency (second index).


        Returns
        -------

        op : FixedTx operator object
            Linear operator for evaluating the model with fixed `s` or `x`.

        """
        if s is not None:
            if x is not None:
                raise ValueError('Cannot create operator for both s and x'
                                  + ' fixed. Please specify only one.')
            return self.FixedTx(s)
        elif x is not None:
            return self.FixedReflectivity(x)
        else:
            raise ValueError('Must specify either s or x')

    def FixedTx(self, s):
        """Create operator for model with fixed transmitted signal `s`.


        Parameters
        ----------

        s : 1-D ndarray
            Real or complex values giving the transmitted radar signal.


        Returns
        -------

        op : FixedTx operator object
            Linear operator for evaluating the model with fixed `s`.

        """
        raise NotImplementedError

    def FixedReflectivity(self, x):
        """Create operator for model with fixed reflectivities `x`.


        Parameters
        ----------

        x : 2-D ndarray
            Complex values giving the reflectivity of the radar target scene
            as a function of delay (first index) and frequency (second index).


        Returns
        -------

        op : FixedTx operator object
            Linear operator for evaluating the model with fixed `x`.

        """
        raise NotImplementedError

class FixedTx(LinearOperator):
    """Point grid radar model operator for a fixed transmitted signal.

    .. automethod:: __init__


    Attributes
    ----------

    delays : 1-D ndarray
        Delay index for the first axis of the target scene reflectivities
        (input to the forward operation and output of the adjoint operation).
        This gives the number of samples by which each reflectivity sample is
        delayed relative to the beginning of the transmitted signal. To get
        the reflectivities with the same delay as the input, index `x` with
        ``[delays >= 0]``.

    freqs : 1-D ndarray
        Normalized frequency index for the second axis of the target scene
        reflectivities (input to the forward operation and output of the
        adjoint operation). This is equivalent to ``np.fft.fftfreq(N, d=1.0)``
        where `N` is the length of the second axis of `x`. To find the Doppler
        frequencies, multiply by the sampling frequency (divide by
        sampling period).

    model : PointGridBase object
        Point grid radar model object that provides signal dimensions and
        precisions for the calculations.

    R : int
        Undersampling ratio, the sampling rate of the transmitted signal
        over the sampling rate of the measured signal.

    s : 1-D ndarray
        Real or complex values giving the transmitted radar signal.

    """
    def __init__(self, s, model):
        """Create point grid radar model operator for fixed TX signal `s`.


        Parameters
        ----------

        s : 1-D ndarray
            Real or complex values giving the transmitted radar signal.

        model : PointGridBase object
            Point grid radar model object to specialize with fixed `s`.

        """
        self.s = s

        self._init_from_model(model)

        inshape = model.xshape
        indtype = model.xydtype
        outshape = model.yshape
        outdtype = model.xydtype

        super(FixedTx, self).__init__(
            inshape, indtype, outshape, outdtype,
        )

    def _init_from_model(self, model):
        self.model = model

        self.delays = model.delays
        self.freqs = model.freqs
        self.R = model.R
        self._fft = model._fft
        self._ifft = model._ifft

class FixedReflectivity(LinearOperator):
    """Point grid radar model operator for a fixed target scene reflectivity.

    .. automethod:: __init__


    Attributes
    ----------

    model : PointGridBase object
        Point grid radar model object that provides signal dimensions and
        precisions for the calculations.

    R : int
        Undersampling ratio, the sampling rate of the transmitted signal
        over the sampling rate of the measured signal.

    x : 2-D ndarray
        Complex values giving the reflectivity of the radar target scene
        as a function of delay (first index) and frequency (second index).

    """
    def __init__(self, x, model):
        """Create point grid radar model operator for fixed reflectivity `x`.


        Parameters
        ----------

        x : 2-D ndarray
            Complex values giving the reflectivity of the radar target scene
            as a function of delay (first index) and frequency (second index).

        model : PointGridBase object
            Point grid radar model object to specialize with fixed `x`.

        """
        self.x = x

        self._init_from_model(model)

        self._xtime = reflectivity_freq2time(x, self._ifft)

        inshape = model.sshape
        indtype = model.sdtype
        outshape = model.yshape
        outdtype = model.xydtype

        super(FixedReflectivity, self).__init__(
            inshape, indtype, outshape, outdtype,
        )

    def _init_from_model(self, model):
        self.model = model

        self.R = model.R
        self._fft = model._fft
        self._ifft = model._ifft


# ****************************************************************************
# *************** TX-referenced Model Class and Operators ********************
# ****************************************************************************
class TxRef(PointGridBase):
    """
    See Also
    --------

    txref_forward : Forward operation.
    txref_adjoint_x : Adjoint operation w.r.t. reflectivities.
    txref_adjoint_s : Adjoint operation w.r.t. transmitted signal.

    """
    __doc__ = PointGridBase.__doc__ + __doc__

    def __init__(self, L, M, N, R=1, precision=np.double):
        # if N < L, we would need to take FFT with nfft >= L so we don't lose
        # data, then subsample to get our N points that we desire
        # this is a lot of extra work just to throw a lot of it away,
        # so don't allow this case (see RxRef for how it would work)
        if N < L:
            raise ValueError('N < L does not result in faster'
                             + ' computation, so it is not allowed. Choose'
                             + ' N >= L.')

        super(TxRef, self).__init__(
            L, M, N, R, precision,
        )
    __init__.__doc__ = PointGridBase.__init__.__doc__

    def FixedTx(self, s):
        return TxRefFixedTx(s, self)
    FixedTx.__doc__ = PointGridBase.FixedTx.__doc__

    def FixedReflectivity(self, x):
        return TxRefFixedReflectivity(x, self)
    FixedReflectivity.__doc__ = PointGridBase.FixedReflectivity.__doc__

    def forward(self, s, x, y=None):
        if y is None:
            y = np.empty(self.yshape, self.xydtype)
        return txref_forward(s, x, self._ifft, y)
    forward.__doc__ = txref_forward.__doc__

    def adjoint_x(self, y, s, x=None):
        if x is None:
            x = np.empty(self.xshape, self.xydtype)
        return txref_adjoint_x(y, s, self._fft, x)
    adjoint_x.__doc__ = txref_adjoint_x.__doc__

    def adjoint_s(self, y, x, s=None):
        raise NotImplementedError
    #adjoint_s.__doc__ = txref_adjoint_s.__doc__

# indicate that default pointgrid model should be the tx-referenced one
# by assigning it to the generic name
PointGrid = TxRef

class TxRefFixedTx(FixedTx):
    """
    See Also
    --------

    txref_forward : Forward operation.
    txref_adjoint_x : Adjoint operation.

    """
    __doc__ = FixedTx.__doc__ + __doc__

    def _forward(self, x, y):
        return txref_forward(self.s, x, self._ifft, y)

    def _adjoint(self, y, x):
        return txref_adjoint_x(y, self.s, self._fft, x)

class TxRefFixedReflectivity(FixedReflectivity):
    """
    See Also
    --------

    txref_forward : Forward operation.
    txref_adjoint_s : Adjoint operation.

    """
    __doc__ = FixedReflectivity.__doc__ + __doc__

    def _forward(self, s, y):
        return txref_forward_delaytime(s, self._xtime, y)

    def _adjoint(self, y, s):
        raise NotImplementedError


# ****************************************************************************
# *************** RX-referenced Model Class and Operators ********************
# ****************************************************************************
class RxRef(PointGridBase):
    """
    nfft : int
        Length of the FFT used to calculate the adjoint operation w.r.t. `x`.
        `nfft` is chosen as the smallest integer multiple of `N` that is
        greater than `M`.

    step : int
        Frequency subsampling stepsize, the ratio of `nfft` to `N`.


    See Also
    --------

    txref_forward : Forward operation.
    txref_adjoint_x : Adjoint operation w.r.t. reflectivities.
    txref_adjoint_s : Adjoint operation w.r.t. transmitted signal.

    """
    __doc__ = PointGridBase.__doc__ + __doc__

    def __init__(self, L, M, N, R=1, precision=np.double):
        # when N < M, need to take FFT with nfft >= M so we don't lose data
        # then subsample to get our N points that we desire
        self.step = M // N + 1
        self.nfft = N*self.step

        super(RxRef, self).__init__(
            L, M, N, R, precision,
        )

    __init__.__doc__ = PointGridBase.__init__.__doc__

    def _create_fft_plan(self):
        delaymult_out = pyfftw.n_byte_align(
            np.zeros((self.P, self.nfft), self.xydtype),
            pyfftw.simd_alignment
        )
        fft_out = pyfftw.n_byte_align(
            np.zeros_like(delaymult_out),
            pyfftw.simd_alignment
        )
        fft = pyfftw.FFTW(delaymult_out, fft_out)

        return fft

    def FixedTx(self, s):
        return RxRefFixedTx(s, self)
    FixedTx.__doc__ = PointGridBase.FixedTx.__doc__

    def FixedReflectivity(self, x):
        return RxRefFixedReflectivity(x, self)
    FixedReflectivity.__doc__ = PointGridBase.FixedReflectivity.__doc__

    def forward(self, s, x, y=None):
        if y is None:
            y = np.empty(self.yshape, self.xydtype)
        return rxref_forward(s, x, self._ifft, y)
    forward.__doc__ = rxref_forward.__doc__

    def adjoint_x(self, y, s, x=None):
        if x is None:
            x = np.empty(self.xshape, self.xydtype)
        return rxref_adjoint_x(y, s, self._fft, x)
    adjoint_x.__doc__ = rxref_adjoint_x.__doc__

    def adjoint_s(self, y, x, s=None):
        raise NotImplementedError
    #adjoint_s.__doc__ = rxref_adjoint_s.__doc__

class RxRefFixedTx(FixedTx):
    """
    See Also
    --------

    rxref_forward : Forward operation.
    rxref_adjoint_x : Adjoint operation.

    """
    __doc__ = FixedTx.__doc__ + __doc__

    def __init__(self, s, model):
        # specialize init for RxRef because inshape is different
        self.s = s

        self._init_from_model(model)

        inshape = model.xshape
        indtype = model.xydtype
        outshape = model.yshape
        outdtype = model.xydtype

        super(FixedTx, self).__init__(
            inshape, indtype, outshape, outdtype,
        )

    def _forward(self, x, y):
        return rxref_forward(self.s, x, self._ifft, y)

    def _adjoint(self, y, x):
        return rxref_adjoint_x(y, self.s, self._fft, x)

class RxRefFixedReflectivity(FixedReflectivity):
    """
    See Also
    --------

    rxref_forward : Forward operation.
    rxref_adjoint_s : Adjoint operation.

    """
    __doc__ = FixedReflectivity.__doc__ + __doc__

    def _forward(self, s, y):
        return rxref_forward_delaytime(s, self._xtime, y)

    def _adjoint(self, y, s):
        raise NotImplementedError
