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

from __future__ import division
import numpy as np
import scipy.sparse as sparse
import pyfftw
import multiprocessing
import timeit
from collections import OrderedDict

import point_forward
import point_adjoint

#__all__ = []

_THREADS = multiprocessing.cpu_count()

def time_models(mlist, x, number=100):
    times = []
    for model in mlist:
        timer = timeit.Timer(lambda: model(x))
        times.append(min(timer.repeat(repeat=3, number=number)))
    
    return times

def measure_forward(s, N, M, R=1, 
                    number=100, disp=True, meas_all=True):
    mlist = [point_forward.CodeFreqSparse(s, N, M, R),
             point_forward.FreqCodeCython(s, N, M, R)]
    if meas_all:
        mlist.extend([point_forward.FreqCodeStrided(s, N, M, R),
                      point_forward.DirectSumCython(s, N, M, R)])
    xshape = (N, R*M)
    x = np.empty(xshape, np.result_type(s.dtype, np.complex64))
    x.real = 2*np.random.rand(*xshape) - 1
    x.imag = 2*np.random.rand(*xshape) - 1
    times = time_models(mlist, x, number)
    
    # sort in order of times
    tups = zip(times, mlist)
    tups.sort()
    
    if disp:
        for time, model in tups:
            print(model.func_name + ': {0} s per call'.format(time/number))
    
    times, mlist = zip(*tups)
    return times, mlist

def measure_adjoint(s, N, M, R=1, 
                    number=100, disp=True, meas_all=True):
    mlist = [point_adjoint.FreqCodeSparse(s, N, M, R),
             point_adjoint.CodeFreqCython(s, N, M, R),
             point_adjoint.CodeFreq2Cython(s, N, M, R)]
    if meas_all:
        mlist.extend([point_adjoint.CodeFreqStrided(s, N, M, R),
                      point_adjoint.CodeFreq2Strided(s, N, M, R),
                      point_adjoint.DirectSumCython(s, N, M, R),
                      point_adjoint.DirectSumNumba(s, N, M, R)])
    y = np.empty(M, np.result_type(s.dtype, np.complex64))
    y.real = 2*np.random.rand(M) - 1
    y.imag = 2*np.random.rand(M) - 1
    times = time_models(mlist, y, number)
    
    # sort in order of times
    tups = zip(times, mlist)
    tups.sort()
    
    if disp:
        for time, model in tups:
            print(model.func_name + ': {0} s per call'.format(time/number))
    
    times, mlist = zip(*tups)
    return times, mlist

def Forward(s, N, M, R=1, measure=True):
    if measure is True:
        times, mlist = measure_forward(s, N, M, R, 
                                       number=10, disp=False, meas_all=False)
        model = mlist[np.argmin(times)]
    else:
        model = point_adjoint.FreqCodeCython(s, N, M, R)
    
    return model

def Adjoint(s, N, M, R=1, measure=True):
    if measure is True:
        times, mlist = measure_adjoint(s, N, M, R,
                                       number=10, disp=False, meas_all=False)
        model = mlist[np.argmin(times)]
    else:
        model = point_adjoint.CodeFreqStrided(s, N, M, R)
    
    return model
