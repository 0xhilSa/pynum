#!/bin/sh

rm -rf build/ dist/ pynum.egg-info/
python3 setup.py sdist bdist_wheel
pip uninstall -y pynum
pip install dist/*whl
