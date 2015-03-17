#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import unittest
import itertools

from radarmodel import point_forward
from radarmodel import point_forward_alt

from radarmodel.util import get_random_uniform, get_random_oncircle

def check_implementation(models, L, N, M, R, sdtype):
    s = get_random_oncircle((L,), sdtype)

    models = [m(s, N, M, R) for m in models]

    inshape = models[0].inshape
    indtype = models[0].indtype
    x = get_random_uniform(inshape, indtype)

    err_msg = 'Result of model "{0}" (y) does not match model "{1}" (x)'

    def call():
        # first in list is used for reference
        refmodel = models[0]
        y0 = refmodel(x)
        for model in models[1:]:
            y1 = model(x)
            try:
                np.testing.assert_array_almost_equal_nulp(y0, y1, nulp=10000)
            except AssertionError as e:
                e.args += (err_msg.format(model.func_name, refmodel.func_name),)
                raise

    call.description = 's={0}({1}), x={2}({3}, {4}), R={5}'.format(np.dtype(sdtype).str,
                                                                   L,
                                                                   np.dtype(indtype).str,
                                                                   inshape[0],
                                                                   inshape[1],
                                                                   R)

    return call

def test_forward():
    models = [getattr(point_forward, modelname) for modelname in point_forward.__all__]
    Ls = (13, 13, 13, 13)
    Ns = (13, 64, 27, 8)
    Ms = (37, 37, 10, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)

    np.random.seed(1)

    for (L, N, M), R, sdtype in itertools.product(zip(Ls, Ns, Ms), Rs, sdtypes):
        callable_test = check_implementation(models, L, N, M, R, sdtype)
        callable_test.description = 'test_forward: ' + callable_test.description
        yield callable_test

def test_forward_alt():
    models = [getattr(point_forward_alt, modelname) for modelname in point_forward_alt.__all__]
    Ls = (13, 13, 13, 13)
    Ns = (13, 64, 27, 8)
    Ms = (37, 37, 10, 37)
    Rs = (1, 2, 3)
    sdtypes = (np.float32, np.complex128)

    np.random.seed(2)

    for (L, N, M), R, sdtype in itertools.product(zip(Ls, Ns, Ms), Rs, sdtypes):
        callable_test = check_implementation(models, L, N, M, R, sdtype)
        callable_test.description = 'test_forward_alt: ' + callable_test.description
        yield callable_test

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture','--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)
