#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import os
import codecs
import copy
from setuptools import setup, Extension, Command
import numpy as np

import versioneer

versioneer.VCS = 'git'
versioneer.versionfile_source = 'radarmodel/_version.py'
versioneer.versionfile_build = 'radarmodel/_version.py'
versioneer.tag_prefix = 'v' # tags are like v1.2.0
versioneer.parentdir_prefix = 'radarmodel-' # dirname like 'radarmodel-1.2.0'

try:
    from Cython.Build import cythonize
    from Cython.Compiler.Options import parse_directive_list
except ImportError:
    HAS_CYTHON = False
else:
    HAS_CYTHON = True

def no_cythonize(extensions, **_ignore):
    dupextensions = copy.deepcopy(extensions)
    for extension in dupextensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return dupextensions

# regular extension modules
ext_modules = []

# cython extension modules
cython_include_path = [] # include for cimport, different from compile include
ext_cython = [Extension('radarmodel.libpoint_forward',
                        sources=['radarmodel/libpoint_forward.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                        extra_link_args=['-O3', '-ffast-math', '-fopenmp']),
              Extension('radarmodel.libpoint_adjoint',
                        sources=['radarmodel/libpoint_adjoint.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                        extra_link_args=['-O3', '-ffast-math', '-fopenmp']),
              Extension('radarmodel.libpoint_forward_alt',
                        sources=['radarmodel/libpoint_forward_alt.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                        extra_link_args=['-O3', '-ffast-math', '-fopenmp']),
              Extension('radarmodel.libpoint_adjoint_alt',
                        sources=['radarmodel/libpoint_adjoint_alt.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                        extra_link_args=['-O3', '-ffast-math', '-fopenmp'])]
# add C-files from cython modules to extension modules
ext_modules.extend(no_cythonize(ext_cython))

# custom setup.py commands
cmdclass = versioneer.get_cmdclass()

if HAS_CYTHON:
    class CythonCommand(Command):
        """Distutils command to cythonize source files."""

        description = "compile Cython code to C code"

        user_options = [('annotate', 'a', 'Produce a colorized HTML version of the source.'),
                        ('directive=', 'X', 'Overrides a compiler directive.'),
                        ('timestamps', 't', 'Only compile newer source files.')]

        def initialize_options(self):
            self.annotate = False
            self.directive = ''
            self.timestamps = False

        def finalize_options(self):
            self.directive = parse_directive_list(self.directive)

        def run(self):
            cythonize(ext_cython,
                      include_path=cython_include_path,
                      force=(not self.timestamps),
                      annotate=self.annotate,
                      compiler_directives=self.directive)

    cmdclass['cython'] = CythonCommand

# Get the long description from the relevant file
# Use codecs.open for Python 2 compatibility
with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='radarmodel',
    version=versioneer.get_version(),
    description='Radar Modeling',
    long_description=long_description,

    url='http://github.com/ryanvolz/radarmodel',

    author='Ryan Volz',
    author_email='ryan.volz@gmail.com',

    license='BSD 3-Clause ("BSD New")',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
    ],

    keywords='radar model',

    packages=['radarmodel'],
    setup_requires=['numpy'],
    install_requires=['numba<=0.11.1', 'numpy', 'pyFFTW', 'scipy'],
    extras_require={
        'develop': ['Cython>=0.17', 'nose'],
    },
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
