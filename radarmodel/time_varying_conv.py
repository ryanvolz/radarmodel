#-----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from numba import jit

__all__ = ['tvconv_by_input', 'tvconv_by_output']

@jit(nopython=True, nogil=True)
def tvconv_by_input_prealloc(s, x, y):
    """tvconv_by_input for pre-allocated output. `y` must be zeroed.

    See :func:`tvconv_by_input` for more information.

    """
    L = len(s)
    M = len(y)
    P = x.shape[0]
    N = x.shape[1]
    R = (P - L)//(M - 1)
    # s obviously can be iterated over [0, L-1]
    # for x, we require that its first dimension is of length P, where
    # P = R*M + L - R. Then this operation is entirely within bounds.
    for l in range(L):
        l_mod_N = l % N # modulus calculation is slow, keep it in outer loop
        for m in range(M):
            y[m] += s[l]*x[R*m - l + L - 1, l_mod_N]

    return y

def tvconv_by_input(s, x, R=1):
    r"""Time-varying convolution between a 1-D sequence and a 2-D kernel.

    The 1-D sequence represents input into a linear time-varying system, and
    the 2-D kernel specifies the system's impulse response as a function of
    *input index* and delay.

    This function implements the equation:

    .. math::

        y[m] = \sum_l ( s[l] x[R m - l + L - 1, l] )

    or equivalently:

    .. math::

        y[m] = \sum_p ( s[R m - p + L - 1] x[p, R m - p + L - 1] )

    for time sequence `s`, system impulse response `x`, downsampling ratio `R`,
    and `L` = len(`s`).


    Parameters
    ----------

    s : 1-D ndarray of length `L`
        Input sequence, indexed by time. :math:`s[l]` gives the input at time
        index `l`.

    x : 2-D ndarray of size (`P`, `N`)
        System impulse response parametrized by input index (time) and delay.
        :math:`x[p, l]` gives the system's output after a delay of `p` time
        steps following an impulse at the input index `l`.

        The length `P` of the delay index determines the length `M` of the
        output such that :math:`R M + L - R = P`. If the length `N` of the first
        index does not match the length `L` of the input index, the entries of
        `x` are either truncated or periodically repeated, as necessary.

    R : int
        Downsampling ratio of the output index relative to the input index.
        For a ratio `R`, the input sequence is sampled at `R` times the rate of
        the output sequence and system kernel's first dimension.


    Returns
    -------

    y : 1-D ndarray of length `M` (see above for how `M` is determined)
        Output sequence, indexed by time *from the end of the input sequence*.
        In other words, the output :math:`y[0]` occurs at the same time as the
        input :math:`s[L - 1]`.


    See Also
    --------

    tvconv_by_output : Similar, but for an output-index-parametrized kernel.


    Notes
    -----

    The output index is offset in time from the input index by the factor
    :math:`L + 1` as a matter of convention in order to simplify calculation by
    eliminating negative indices and their implicit zero-values.

    """
    L = len(s)
    P = x.shape[0]
    M = (P - L) // R + 1
    outdtype = np.result_type(s.dtype, x.dtype)
    y = np.zeros(M, dtype=outdtype)
    return tvconv_by_input_prealloc(s, x, y)

@jit(nopython=True, nogil=True)
def tvconv_by_output_prealloc(s, x, y):
    """tvconv_by_output for pre-allocated output. `y` must be zeroed.

    See :func:`tvconv_by_output` for more information.

    """
    L = len(s)
    M = len(y)
    P = x.shape[0]
    N = x.shape[1]
    R = (P - L)//(M - 1)
    for m in range(M):
        m_mod_N = m % N # modulus calculation is slow, keep it in outer loop
        # s obviously can be iterated over [0, L-1]
        # for x, we require that its first dimension is of length P, where
        # P = R*M + L - R. Then this operation is entirely within bounds.
        for l in range(L):
            y[m] += s[l]*x[R*m - l + L - 1, m_mod_N]

    return y

def tvconv_by_output(s, x, R=1):
    r"""Time-varying convolution between a 1-D sequence and a 2-D kernel.

    The 1-D sequence represents input into a linear time-varying system, and
    the 2-D kernel specifies the system's impulse response as a function of
    *output index* and delay.

    This function implements the equation:

    .. math::

        y[m] = \sum_p ( s[R m - p + L - 1] x[p, m] )

    or equivalently:

    .. math::

        y[m] = \sum_l ( s[l] x[R m - l + L - 1, m] )

    for time sequence `s`, system impulse response `x`, downsampling ratio `R`,
    and `L` = len(`s`).


    Parameters
    ----------

    s : 1-D ndarray of length `L`
        Input sequence, indexed by time. :math:`s[l]` gives the input at time
        index `l`.

    x : 2-D ndarray of size (`P`, `N`)
        System impulse response parametrized by output index (time) and delay.
        :math:`x[p, m]` gives the system's output at index `m` in response to
        an impulse at the input index :math:`l = R m - p + L - 1`, i.e. `p` time
        steps prior to the output.

        The length `P` of the delay index determines the length `M` of the
        output such that :math:`R M + L - R = P`. If the length `N` of the first
        index does not match `M`, the entries of `x` are either truncated or
        periodically repeated, as necessary.

    R : int
        Downsampling ratio of the output index relative to the input index.
        For a ratio `R`, the input sequence is sampled at `R` times the rate of
        the output sequence and system kernel's first dimension.


    Returns
    -------

    y : 1-D ndarray of length `M` (see above for how `M` is determined)
        Output sequence, indexed by time *from the end of the input sequence*.
        In other words, the output :math:`y[0]` occurs at the same time as the
        input :math:`s[L - 1]`.


    See Also
    --------

    tvconv_by_input : Similar, but for an input-index-parametrized kernel.


    Notes
    -----

    The output index is offset in time from the input index by the factor
    :math:`L + 1` as a matter of convention in order to simplify calculation by
    eliminating negative indices and their implicit zero-values.

    """
    L = len(s)
    P = x.shape[0]
    M = (P - L) // R + 1
    outdtype = np.result_type(s.dtype, x.dtype)
    y = np.zeros(M, dtype=outdtype)
    return tvconv_by_output_prealloc(s, x, y)
