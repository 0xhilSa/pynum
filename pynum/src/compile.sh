nvcc --compiler-options '-fPIC' -std=c++17 -shared \
    $(python3 -m pybind11 --includes) \
    cu_manager.c -o cu_manager$(python3-config --extension-suffix) \
    -lcudart

