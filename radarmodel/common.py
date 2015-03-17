#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

def model_dec(inshape, indtype, outshape, outdtype):
    def decor(fun):
        fun.inshape = inshape
        fun.indtype = indtype
        fun.outshape = outshape
        fun.outdtype = outdtype

        return fun
    return decor
