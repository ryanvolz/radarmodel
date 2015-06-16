$PYTHON setup.py cython
$PYTHON setup.py build
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
# automatically get version from _version.py
cd $PKG_NAME
$PYTHON -c "from _version import get_versions; print(get_versions()['version'])" > "$SRC_DIR/__conda_version__.txt"
