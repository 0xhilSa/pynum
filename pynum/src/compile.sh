nvcc --compiler-options '-fPIC' -std=c++17 -shared \
  $(python3 -m pybind11 --includes) \
  cuda_stream.c -o cuda_stream$(python3-config --extension-suffix) \
  -lcudart
