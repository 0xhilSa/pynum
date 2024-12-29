#!/bin/sh

rm -rf build/ dist/ pynum.egg-info/
python3 setup.py sdist bdist_wheel
pip uninstall -y pynum
pip install dist/*whl
#mkdir build && cd build 
#cmake ..
#cd ..
#python3 -m build
#pip uninstall -y pynum
#pip install dist/*whl



