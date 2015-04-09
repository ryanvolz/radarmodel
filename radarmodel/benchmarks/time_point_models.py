# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------


import numpy as np

from radarmodel import pointgrid
from radarmodel.util import get_random_uniform, get_random_oncircle

class ModelSuite:
    param_names = ['params', 'model']
    params = ([
        dict(L=13, N=16, M=50, R=1, precision=np.dtype(np.float32)),
        dict(L=51, N=64, M=500, R=1, precision=np.dtype(np.float32)),
        dict(L=51, N=64, M=500, R=1, precision=np.dtype(np.float64)),
        dict(L=51, N=64, M=500, R=3, precision=np.dtype(np.float32)),
        dict(L=101, N=128, M=2000, R=1, precision=np.dtype(np.float32)),
        dict(L=101, N=512, M=2000, R=1, precision=np.dtype(np.float32)),
    ],
    ['TxRef', 'RxRef'])

    def setup(self, params, modelname):
        cls = getattr(pointgrid, modelname)
        model = cls(**params)

        s = get_random_oncircle(model.sshape, model.sdtype)
        x = get_random_uniform(model.xshape, model.xydtype)
        y = get_random_uniform(model.yshape, model.xydtype)

        y_out = model.forward(s, x)
        x_out = model.adjoint_x(y, s)

        self.model = model
        self.s = s
        self.x = x
        self.x_out = x_out
        self.y = y
        self.y_out = y_out

    def time_forward(self, *args):
        self.model.forward(self.s, self.x, self.y_out)

    def time_adjoint_x(self, *args):
        self.model.adjoint_x(self.y, self.s, self.x_out)
