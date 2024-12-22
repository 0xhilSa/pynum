# PyNum (WIP)

***a small python library for 1D and 2D arrays with GPU supports***

## TODO
- ops between dtypes
- check whether to use pyopencl or go with C++ CUDA programming
- vector(1D array) and its ops for CPU and GPU

## Prerequisites
Before building the library, ensure the following dependencies are installed
- cmake
- pybind11

## Installation and Usage
To clone, build, and install the library, follow these steps:
```bash
# clone the repository
git clone "https://github.com/0xhilSa/pynum"

# navigate to the project directory
cd pynum

# create a build directory and navigate into it
mkdir build && cd build

# generate build files using CMake
cmake ..

# return to the main directory
cd ..

# build the Python package and install it
python3 -m build
pip install dist/pynum-0.0.1-cp310-cp310-linux_x86_64.whl
```