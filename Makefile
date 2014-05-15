# simple makefile to simplify repetitive build env management tasks under posix

# allow specifying python version on command line to ease testing with
# different versions, e.g. $ PYTHON=/usr/bin/python3 make test
PYTHON ?= python
NOSETESTS ?= nosetests

all: clean inplace test

clean:
	-$(PYTHON) setup.py clean

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

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext --inplace

test: test_code

test_code: in
	$(NOSETESTS) -s -v

test_coverage:
	-rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage

# doc: inplace
# 	$(MAKE) -C doc html
