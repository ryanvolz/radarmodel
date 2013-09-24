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
import unittest
import itertools

from radarmodel import point_forward
from radarmodel import point_forward_alt

from util import get_random_uniform, get_random_oncircle

def check_implementation(models, L, N, M, R, sdtype, xdtype):
    s = get_random_oncircle((L,), sdtype)
    x = get_random_uniform((N, R*M), xdtype)
    
    models = [m(s, N, M, R) for m in models]
    
    err_msg = 'Result of model "{0}" (y) does not match model "{1}" (x)'
    
    def call():
        # first in list is used for reference
        refmodel = models[0]
        y0 = refmodel(x)
        for model in models[1:]:
            y1 = model(x)
            np.testing.assert_array_almost_equal(y0, y1, err_msg=err_msg.format(model.func_name, refmodel.func_name))
    
    call.description = 's={0}({1}), x={2}({3}, {4}), R={5}'.format(np.dtype(sdtype).str,
                                                                   L,
                                                                   np.dtype(xdtype).str,
                                                                   N,
                                                                   R*M,
                                                                   R)
    
    return call

def test_forward():
    models = [getattr(point_forward, modelname) for modelname in point_forward.__all__]
    Ls = (13, 13, 13)
    Ns = (13, 64, 8)
    Ms = (37, 37, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)
    xdtypes = (np.complex64, np.complex128)
    
    for (L, N, M), R, (sdtype, xdtype) in itertools.product(zip(Ls, Ns, Ms), Rs, zip(sdtypes, xdtypes)):
        callable_test = check_implementation(models, L, N, M, R, sdtype, xdtype)
        callable_test.description = 'test_forward: ' + callable_test.description
        yield callable_test

def test_forward_alt():
    models = [getattr(point_forward_alt, modelname) for modelname in point_forward_alt.__all__]
    Ls = (13, 13, 13)
    Ns = (13, 64, 8)
    Ms = (37, 37, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)
    xdtypes = (np.complex64, np.complex128)
    
    for (L, N, M), R, (sdtype, xdtype) in itertools.product(zip(Ls, Ns, Ms), Rs, zip(sdtypes, xdtypes)):
        callable_test = check_implementation(models, L, N, M, R, sdtype, xdtype)
        callable_test.description = 'test_forward_alt: ' + callable_test.description
        yield callable_test

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture','--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)