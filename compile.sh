#!/bin/sh

rm -rf build/ dist/ pynum.egg-info/
mkdir build && cd build 
cmake ..
cd ..
python3 -m build
pip uninstall -y pynum
pip install dist/*whl

