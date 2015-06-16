#-----------------------------------------------------------------------------
# Copyright (c) 2015, 'radarmodel' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from setuptools import setup, find_packages
# to use a consistent encoding
from codecs import open
from os import path
import copy
import numpy as np

import versioneer

# custom setup.py commands
cmdclass = versioneer.get_cmdclass()

# add nose and sphinx commands since we depend on them but they are not always
# automatically available (e.g. when using conda versions of these packages)
try:
    from nose.commands import nosetests
except ImportError:
    pass
else:
    cmdclass['nosetests'] = nosetests
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    pass
else:
    cmdclass['build_sphinx'] = BuildDoc

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='radarmodel',
    version=versioneer.get_version(),
    description='Mathematical radar models useful for inverting radar measurements',
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
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ],

    keywords='radar model inverse inversion',

    packages=find_packages(),
    setup_requires=['numpy'],
    install_requires=['numba', 'numpy', 'pyFFTW', 'scipy'],
    extras_require={
        'develop': ['flake8', 'nose', 'pylint', 'twine', 'wheel'],
        'doc': ['numpydoc', 'sphinx'],
    },
    cmdclass=cmdclass,
)
