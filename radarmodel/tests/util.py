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