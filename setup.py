#!/usr/bin/env python

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

import sys
from distutils.core import setup, Extension, Command
import numpy as np

try:
    from Cython.Compiler.Main import compile, CompilationResultSet
except ImportError:
    cython = False
else:
    cython = True

cython_sources = [('radarmodel/libpoint_forward.pyx',
                            [np.get_include(), 'radarmodel/include']),
                  ('radarmodel/libpoint_adjoint.pyx',
                            [np.get_include(), 'radarmodel/include'])]

cmdclass = dict()

if cython:
    class CythonCommand(Command):
        """Distutils command to cythonize source files."""
        
        description = "compile Cython code to C code"
        
        user_options = [('annotate', 'a', 'Produce a colorized HTML version of the source.'),
                        ('timestamps', 't', 'Only compile newer source files')]
        
        def initialize_options(self):
            self.annotate = False
            self.timestamps = False
        
        def finalize_options(self):
            pass
        
        def run(self):
            results = CompilationResultSet()
            
            for source, include_path in cython_sources:
                res = compile([source],
                              include_path=include_path,
                              verbose=True,
                              timestamps=self.timestamps,
                              annotate=self.annotate)
                if res:
                    results.add(source, res.values()[0])
            
            if results.num_errors > 0:
                sys.stderr.write('Cython compilation failed!')

    cmdclass['cython'] = CythonCommand

ext_modules = [Extension('radarmodel.libpoint_forward',
                         sources=['radarmodel/libpoint_forward.c'],
                         include_dirs=[np.get_include(), 'radarmodel/include'],
                         extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                         extra_link_args=['-O3', '-ffast-math', '-fopenmp']),
               Extension('radarmodel.libpoint_adjoint',
                         sources=['radarmodel/libpoint_adjoint.c'],
                         include_dirs=[np.get_include(), 'radarmodel/include'],
                         extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                         extra_link_args=['-O3', '-ffast-math', '-fopenmp'])]

setup(name='radarmodel',
      version='0.1-dev',
      maintainer='Ryan Volz',
      maintainer_email='ryan.volz@gmail.com',
      url='http://sess.stanford.edu',
      description='Radar Modeling',
      long_description='',
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Operating System :: OS Independent',
                   'Programming Language :: Cython',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Topic :: Scientific/Engineering'],
      packages=['radarmodel'],
      package_data={'radarmodel': ['include/*']},
      data_files=[('', list(zip(*cython_sources)[0]))],
      cmdclass=cmdclass,
      ext_modules=ext_modules)
