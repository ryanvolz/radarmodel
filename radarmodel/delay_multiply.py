#-----------------------------------------------------------------------------
# Copyright (c) 2015, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from numba import jit

__all__ = ['delaymult_like_arg1', 'delaymult_like_arg2']

@jit(nopython=True, nogil=True)
def delaymult_like_arg1_prealloc(a, v, R, out):
    """delaymult_like_arg1 for pre-allocated output `out`.

    See :func:`delaymult_like_arg1` for more information.

    """
    M = len(a)
    L = len(v)
    for l in range(L):
        for m in range(M):
            out[R*m - l + L - 1, m] = a[m]*np.conj(v[l])

    return out

def delaymult_like_arg1(a, v, R=1):
    r"""Multiply two 1-D sequences at all overlapping relative delays.

    The second sequence `v` is conjugated and multiplied element-by-element
    with the first sequence at all relative delays where the two can overlap,
    with zero-padding as necessary. This function computes

    .. math::

        x[k, m] = a[m] v^*[R m - k + L - 1]

    or equivalently

    .. math::

        x[R m - l + L - 1, m] = a[m] v^*[l]

    for sequences `a` and `v`, undersampling ratio `R`, and `L` = len(`v`).

    The calculation is like a full cross-correlation of the two
    sequences, except that the result is not summed for each delay. Instead, the
    output is two-dimensional. The first dimension indexes the relative
    delay, and the second dimension indexes the values of the *first* sequence
    `a` multiplied by the appropriately delayed values of the second sequence
    `v`. Summing the output over the second axis will give a result equal to
    :func:`numpy.correlate`(`a`, `v`, mode='full').


    Parameters
    ----------

    a : 1-D ndarray of length `M`
        Input sequence.

    v : 1-D ndarray of length `L`
        Input sequence.

    R : int
        Undersampling ratio of `a` relative to `v`. For a ratio `R`, the
        sampling rate of `v` is taken to be `R` times the sampling rate of `a`,
        so subsequent samples of `a` are correlated with samples of `v` that are
        `R` steps apart.


    Returns
    -------

    x : 2-D ndarray of size (`P`, `L`)
        Result :math:`x[k, l]` of the element-by-element multiplication of the
        input sequences for each relative delay `k`, indexed along the second
        axis corresponding to the *first* sequence :math:`a[m]`. The length `P`
        of the first dimension satisfies :math:`P = R M + L - R`.


    See Also
    --------

    delaymult_like_arg2 : Same, but result indexed following second argument.

    numpy.correlate : Similar calculation, except summed over the second axis.


    Notes
    -----

    This formulation is useful in instances where a correlation-like
    multiplication is required as part of a larger calculation that involves
    more than just summing the result. For example, computing a DFT over the
    output's second axis produces a frequency-shifted filter derived from `v`
    and applied to `a`.

    """
    M = len(a)
    L = len(v)
    P = R*M + L - R
    out = np.zeros((P, M), dtype=a.dtype)
    return delaymult_like_arg1_prealloc(a, v, R, out)

@jit(nopython=True, nogil=True)
def delaymult_like_arg2_prealloc(a, v, R, out):
    """delaymult_like_arg2 for pre-allocated output `out`.

    See :func:`delaymult_like_arg2` for more information.

    """
    M = len(a)
    L = len(v)
    for m in range(M):
        for l in range(L):
            out[R*m - l + L - 1, l] = a[m]*np.conj(v[l])

    return out

def delaymult_like_arg2(a, v, R=1):
    r"""Multiply two 1-D sequences at all overlapping relative delays.

    The second sequence `v` is conjugated and multiplied element-by-element
    with the first sequence at all relative delays where the two can overlap,
    with zero-padding as necessary. This function computes

    .. math::

        x[k, l] = a_R[k + l - (L - 1)] v^*[l]

    or equivalently

    .. math::

        x[R m - l + L - 1, l] = a[m] v^*[l]

    where :math:`a_R` is `a` upsampled by `R` (insert `R`-1 zeros after each
    original element) for sequences `a` and `v`, undersampling ratio `R`, and
    `L` = len(`v`).

    The calculation is like a full cross-correlation of the two
    sequences, except that the result is not summed for each delay. Instead, the
    output is two-dimensional. The first dimension indexes the relative
    delay, and the second dimension indexes the values of the *second* sequence
    `v` multiplied by the appropriately delayed values of the first sequence
    `a`. Summing the output over the second axis will give a result equal to
    :func:`numpy.correlate`(`a`, `v`, mode='full').


    Parameters
    ----------

    a : 1-D ndarray of length `M`
        Input sequence.

    v : 1-D ndarray of length `L`
        Input sequence.

    R : int
        Undersampling ratio of `a` relative to `v`. For a ratio `R`, the
        sampling rate of `v` is taken to be `R` times the sampling rate of `a`,
        so subsequent samples of `a` are correlated with samples of `v` that are
        `R` steps apart.


    Returns
    -------

    x : 2-D ndarray of size (`P`, `L`)
        Result :math:`x[k, l]` of the element-by-element multiplication of the
        input sequences for each relative delay `k`, indexed along the second
        axis corresponding to the *second* sequence :math:`v[l]`. The length `P`
        of the first dimension satisfies :math:`P = R M + L - R`.


    See Also
    --------

    delaymult_like_arg1 : Same, but result indexed following first argument.

    numpy.correlate : Similar calculation, except summed over the second axis.


    Notes
    -----

    This formulation is useful in instances where a correlation-like
    multiplication is required as part of a larger calculation that involves
    more than just summing the result. For example, computing a DFT over the
    output's second axis produces a frequency-shifted filter derived from `v`
    and applied to `a`.

    """
    M = len(a)
    L = len(v)
    P = R*M + L - R
    out = np.zeros((P, L), dtype=y.dtype)
    return delaymult_like_arg2_prealloc(a, v, R, out)
