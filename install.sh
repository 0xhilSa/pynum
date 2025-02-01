#!/bin/sh

rm -rf ./build/ ./dist/ ./pynum.egg-info/
pip uninstall -y pynum
python3 setup.py sdist bdist_wheel
pip install ./dist/*whl
