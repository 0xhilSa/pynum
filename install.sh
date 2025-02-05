#!/bin/sh

[ -d "./build" ] && rm -rf ./build
[ -d "./dist" ] && rm -rf ./dist
[ -d "./pynum.egg-info" ] && rm -rf ./pynum.egg-info
pip uninstall -y pynum

cd ./pynum/csrc/

# compile pycu.cu using NVCC compiler CUDA must be available on your device
nvcc --compiler-options '-fPIC' -std=c++17 -shared \
 $(python3 -m pybind11 --includes) \
 pycu.cu -o pycu$(python3-config --extension-suffix) \
 -lcudart

# compile host.c using GCC compiler
gcc -shared -fPIC -Wall \
  $(python3 -m pybind11 --includes) \
  host.c -o host$(python3-config --extension-suffix) \
  $(python3-config --ldflags)

cd ../..

python3 setup.py sdist bdist_wheel
pip install ./dist/*whl
rm -rf ./build ./dist ./pynum.egg-info
