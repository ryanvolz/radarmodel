#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import scipy.sparse as sparse
import pyfftw
import timeit
from collections import OrderedDict

from . import point_forward
from . import point_adjoint
from . import point_forward_alt
from . import point_adjoint_alt
from . import util

__all__ = ['Adjoint', 'Adjoint_alt', 'Forward', 'Forward_alt', 
           'fastest_adjoint', 'fastest_adjoint_alt', 'fastest_forward', 'fastest_forward_alt',
           'measure_adjoint', 'measure_adjoint_alt', 'measure_forward', 'measure_forward_alt',
           'time_models']

def time_models(mlist, x, number=100):
    times = []
    for model in mlist:
        timer = timeit.Timer(lambda: model(x))
        times.append(min(timer.repeat(repeat=3, number=number)))
    
    return times

def measure_factory(always, others):
    def measure(s, N, M, R=1, number=100, disp=True, meas_all=False):
        """Return (times, models), measured running times of model implementations.
        
    """
        mlist = [mdl(s, N, M, R) for mdl in always]
        if meas_all:
            mlist.extend([mdl(s, N, M, R) for mdl in others])
        
        inshape = mlist[0].inshape
        indtype = mlist[0].indtype
        x = util.get_random_normal(inshape, indtype)
        
        times = time_models(mlist, x, number)
        
        # sort in order of times
        tups = zip(times, mlist)
        tups.sort()
        
        if disp:
            for time, model in tups:
                print(model.func_name + ': {0} s per call'.format(time/number))
        
        times, mlist = zip(*tups)
        return times, mlist
    
    measure.__doc__ += always[0].__doc__
    measure.__doc__ += """
    number: int
        Number of times to run each model for averaging execution time.
    
    disp: bool
        If True, prints time results in addition to returning them.
    
    meas_all: bool
        If False, measure only the models expected to be fastest. 
        If True, measure all models.
    
    """
    
    return measure

def fastest_factory(always, others):
    measurefun = measure_factory(always, others)
    
    def fastest(s, N, M, R=1, number=100, meas_all=False):
        """Return fastest model implementation for the given parameters.
        
    """
        times, mlist = measurefun(s, N, M, R=R, number=number, 
                                  disp=False, meas_all=meas_all)
        return mlist[np.argmin(times)]
    
    fastest.__doc__ += always[0].__doc__
    fastest.__doc__ += """
    number: int
        Number of times to run each model for averaging execution time.
    
    meas_all: bool
        If False, measure only the models expected to be fastest. 
        If True, measure all models.
    
    """
    
    return fastest


measure_forward = measure_factory(
    [point_forward.FreqCodeCython,
     point_forward.FreqCodeStrided],
    [point_forward.CodeFreqSparse,
     point_forward.DirectSumCython]
    )

fastest_forward = fastest_factory(
    [point_forward.FreqCodeCython,
     point_forward.FreqCodeStrided],
    [point_forward.CodeFreqSparse,
     point_forward.DirectSumCython]
    )

Forward = point_forward.FreqCodeCython

measure_forward_alt = measure_factory(
    [point_forward_alt.FreqCodeCython,
     point_forward_alt.FreqCodeStrided],
    []
    )

fastest_forward_alt = fastest_factory(
    [point_forward_alt.FreqCodeCython,
     point_forward_alt.FreqCodeStrided],
    []
    )

Forward_alt = point_forward_alt.FreqCodeCython

measure_adjoint = measure_factory(
    [point_adjoint.CodeFreqCython,
     point_adjoint.CodeFreqStrided],
    [point_adjoint.DirectSumCython,
     point_adjoint.DirectSumNumba,
     point_adjoint.FreqCodeSparse]
    )

fastest_adjoint = fastest_factory(
    [point_adjoint.CodeFreqCython,
     point_adjoint.CodeFreqStrided],
    [point_adjoint.DirectSumCython,
     point_adjoint.DirectSumNumba,
     point_adjoint.FreqCodeSparse]
    )

Adjoint = point_adjoint.CodeFreqStrided

measure_adjoint_alt = measure_factory(
    [point_adjoint_alt.CodeFreqCython,
     point_adjoint_alt.CodeFreqStrided],
    []
    )

fastest_adjoint_alt = fastest_factory(
    [point_adjoint_alt.CodeFreqCython,
     point_adjoint_alt.CodeFreqStrided],
    []
    )

Adjoint_alt = point_adjoint_alt.CodeFreqStrided