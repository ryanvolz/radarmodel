# simple makefile to simplify repetitive build env management tasks under posix

# allow specifying python version on command line to ease testing with
# different versions, e.g. $ PYTHON=/usr/bin/python3 make test
PYTHON ?= python

PACKAGE = radarmodel

.PHONY: all clean clean_build clean_coverage clean_inplace clean_sphinxbuild code_analysis code_check dist distclean doc doc_force in inplace inplace_force pdf test test_code test_coverage

all: clean inplace test

clean:
	-$(PYTHON) setup.py clean
	make -C doc clean

clean_build: clean_egginfo
	-$(PYTHON) setup.py clean --all

clean_coverage:
	-rm -rf coverage .coverage

clean_egginfo:
	-rm -rf "$(PACKAGE).egg-info"

clean_inplace:
	-find . -name '*.py[cdo]' -exec rm {} \;
	-find $(PACKAGE) \( -name '*.dll' -o -name '*.so' \) -exec rm {} \;
	-find $(PACKAGE) -name '*.html' -exec rm {} \;

clean_sphinxbuild:
	-rm -rf build/sphinx

code_analysis:
	-pylint --output-format colorized --extension-pkg-whitelist=numpy $(PACKAGE)

code_check:
	flake8 $(PACKAGE) | grep -v __init__ | grep -v _version
	pylint --errors-only --output-format colorized --extension-pkg-whitelist=numpy $(PACKAGE)

dist: clean_egginfo
	$(PYTHON) setup.py sdist

distclean: clean_build clean_inplace clean_sphinxbuild
	make -C doc distclean

doc: inplace
	make -C doc html

doc_force: inplace
	make -C doc distclean
	make -C doc html

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext --inplace

inplace_force:
	$(PYTHON) setup.py build_ext --inplace --force

pdf:
	$(PYTHON) setup.py build_sphinx --fresh-env --builder latex
	make -C build/sphinx/latex all-pdf

test: test_code

test_code: inplace
	nosetests --nocapture --verbosity=2

test_coverage: inplace clean_coverage
	nosetests --nocapture --verbosity=2 --with-coverage

wheel: clean_egginfo
	$(PYTHON) setup.py bdist_wheel
