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
import itertools

from radarmodel import point_forward, point_adjoint, point_forward_alt, point_adjoint_alt

from util import get_random_normal, get_random_oncircle

def adjointness_error(A, Astar, inshape, indtype, its=100):
    """Check adjointness of A and Astar for 'its' instances of random data.
    
    For random unit-normed x and y, this finds the error in the adjoint 
    identity <Ax, y> == <x, A*y>:
        err = abs( vdot(A(x), y) - vdot(x, Astar(y)) ).
    
    The type and shape of the input to A are specified by inshape and indtype.
    
    Returns a vector of the error magnitudes.
    
    """
    x = get_random_normal(inshape, indtype)
    y = A(x)
    outshape = y.shape
    outdtype = y.dtype
    
    errs = np.zeros(its, dtype=indtype)
    for k in xrange(its):
        x = get_random_normal(inshape, indtype)
        x = x/np.linalg.norm(x)
        y = get_random_normal(outshape, outdtype)
        y = y/np.linalg.norm(y)
        ip_A = np.vdot(A(x), y)
        ip_Astar = np.vdot(x, Astar(y))
        errs[k] = np.abs(ip_A - ip_Astar)
    
    return errs

def check_adjointness(formodel, adjmodel, L, N, M, R, sdtype, xdtype):
    s = get_random_oncircle((L,), sdtype)
    s = s/np.linalg.norm(s)
    
    A = formodel(s, N, M, R)
    Astar = adjmodel(s, N, M, R)
    
    err_msg = '{0} and {1} are not adjoints, with max error of {2}'
    
    def call():
        errs = adjointness_error(A, Astar, (N, R*M), xdtype, its=100)
        np.testing.assert_array_almost_equal(errs, 0, 
            err_msg=err_msg.format(formodel.func_name, adjmodel.func_name, np.max(np.abs(errs))))
    
    call.description = 's={0}({1}), x={2}({3}), N={4}, R={5}'.format(np.dtype(sdtype).str,
                                                                     L,
                                                                     np.dtype(xdtype).str,
                                                                     M,
                                                                     N,
                                                                     R)
    
    return call

def test_adjointness():
    Afun = point_forward.FreqCodeCython
    Astarfun = point_adjoint.CodeFreqCython
    Ls = (13, 13, 13)
    Ns = (13, 64, 8)
    Ms = (37, 37, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)
    xdtypes = (np.complex64, np.complex128)
    
    np.random.seed(1)
    
    for (L, N, M), R, (sdtype, xdtype) in itertools.product(zip(Ls, Ns, Ms), Rs, zip(sdtypes, xdtypes)):
        callable_test = check_adjointness(Afun, Astarfun, L, N, M, R, sdtype, xdtype)
        callable_test.description = 'test_adjointness: ' + callable_test.description
        yield callable_test

def test_adjointness_alt():
    Afun = point_forward_alt.FreqCodeCython
    Astarfun = point_adjoint_alt.CodeFreqCython
    Ls = (13, 13, 13)
    Ns = (13, 64, 8)
    Ms = (37, 37, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)
    xdtypes = (np.complex64, np.complex128)
    
    np.random.seed(2)
    
    for (L, N, M), R, (sdtype, xdtype) in itertools.product(zip(Ls, Ns, Ms), Rs, zip(sdtypes, xdtypes)):
        callable_test = check_adjointness(Afun, Astarfun, L, N, M, R, sdtype, xdtype)
        callable_test.description = 'test_adjointness_alt: ' + callable_test.description
        yield callable_test

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture','--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)