#!/bin/bash

#nvcc --compiler-options '-fPIC' -std=c++17 -shared \
#  $(python3 -m pybind11 --includes) \
#  pycu.cu -o pycu$(python3-config --extension-suffix) \
#  -lcudart

gcc -shared -fPIC -Wall \
  $(python3 -m pybind11 --includes) \
  host.c -o host$(python3-config --extension-suffix) \
  $(python3-config --ldflags)
