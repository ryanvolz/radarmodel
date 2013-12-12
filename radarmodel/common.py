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

def model_dec(inshape, indtype, outshape, outdtype):
    def decor(fun):
        fun.inshape = inshape
        fun.indtype = indtype
        fun.outshape = outshape
        fun.outdtype = outdtype
        
        return fun
    return decor