# simple makefile to simplify repetitive build env management tasks under posix

# allow specifying python version on command line to ease testing with
# different versions, e.g. $ PYTHON=/usr/bin/python3 make test
PYTHON ?= python

all: clean inplace test

clean:
	-$(PYTHON) setup.py clean
	make -C doc clean

clean_build:
	-$(PYTHON) setup.py clean --all

clean_coverage:
	-rm -rf coverage .coverage

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

code_analysis:
	-pylint -i y -f colorized radarmodel

code_check:
	flake8 radarmodel | grep -v __init__ | grep -v _version
	pylint -E -i y -f colorized radarmodel

cython:
	$(PYTHON) setup.py cython --timestamps

cython_annotate:
	$(PYTHON) setup.py cython --annotate

cython_force:
	$(PYTHON) setup.py cython

distclean: clean_build clean_pyc
	make -C doc distclean

doc: inplace
	make -C doc html

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext --inplace

inplace_force:
	$(PYTHON) setup.py build_ext --inplace --force

test: test_code

test_code: cython inplace
	$(PYTHON) setup.py nosetests --nocapture --verbose

test_coverage: cython inplace clean_coverage
	$(PYTHON) setup.py nosetests --nocapture --verbose --with-coverage
