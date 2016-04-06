#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

def get_random_normal(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = np.random.randn(*shape)
    try:
        x.imag = np.random.randn(*shape)
    except TypeError:
        pass
    return x

def get_random_uniform(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = 2*np.random.rand(*shape) - 1
    try:
        x.imag = 2*np.random.rand(*shape) - 1
    except TypeError:
        pass
    return x

def get_random_oncircle(shape, dtype):
    x = np.empty(shape, dtype)
    if np.iscomplexobj(x):
        x[:] = np.exp(2*np.pi*1j*np.random.rand(*shape))
    else:
        x[:] = 2*(np.random.rand(*shape) > 0.5) - 1
    return x
