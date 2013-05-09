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

import point_adjoint

def get_random_uniform(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = 2*np.random.rand(*shape) - 1
    if np.iscomplexobj(x):
        x.imag = 2*np.random.rand(*shape) - 1
    return x

def get_random_oncircle(shape, dtype):
    return np.exp(2*np.pi*1j*np.random.rand(*shape)).astype(dtype)

def check_implementation(L, N, M, R, sdtype, ydtype):
    s = get_random_oncircle((L,), sdtype)
    y = get_random_uniform((M,), ydtype)
    
    # first in list is used for reference
    models = [point_adjoint.DirectSum(s, N, M, R),
              point_adjoint.FreqCodeSparse(s, N, M, R), 
              point_adjoint.CodeFreqStrided(s, N, M, R),
              point_adjoint.DirectSumCython(s, N, M, R),
              point_adjoint.CodeFreqCython(s, N, M, R),
              point_adjoint.DirectSumNumba(s, N, M, R)]
    #models = [point_adjoint.CodeFreq2Strided(s, N, M, R),
              #point_adjoint.CodeFreq2Cython(s, N, M, R)]
    
    err_msg = 'Result of model "{0}" (y) does not match model "{1}" (x)'
    
    refmodel = models[0]
    x0 = refmodel(y)
    for model in models[1:]:
        x1 = model(y)
        np.testing.assert_array_almost_equal(x0, x1, err_msg=err_msg.format(model.func_name, refmodel.func_name))

def test_implementation():
    Ls = (13, 13)
    Ns = (13, 64)
    Ms = (37, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)
    ydtypes = (np.complex64, np.complex128)
    
    for (L, N, M), R, (sdtype, ydtype) in itertools.product(zip(Ls, Ns, Ms), Rs, zip(sdtypes, ydtypes)):
        yield check_implementation, L, N, M, R, sdtype, ydtype

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture','--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)