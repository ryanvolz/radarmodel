# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------


import numpy as np
import timeit

from radarmodel import pointgrid
from radarmodel.util import get_random_uniform, get_random_oncircle

class ModelSuite:
    param_names = ['model', 'params']
    params = (['TxRef', 'RxRef'], [
        dict(L=13, N=16, M=37, R=1, precision=np.double),
    ])
    number = 1000
    repeat = 100

    def setup(self, modelname, params):
        cls = getattr(pointgrid, modelname)
        model = cls(**params)

        s = get_random_oncircle(model.sshape, model.sdtype)
        x = get_random_uniform(model.xshape, model.xydtype)
        y = get_random_uniform(model.yshape, model.xydtype)

        self.model = model
        self.s = s
        self.x = x
        self.y = y

    def time_forward(self):
        self.model.forward(self.s, self.x, self.y)

    def time_adjoint_x(self):
        self.model.adjoint_x(self.y, self.s, self.x)

def main():
    """Main function that runs point model benchmarks when run as a script."""
    pass

if __name__ == '__main__':
    main()
