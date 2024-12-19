cmake_minimum_required(VERSION 3.12)
project(pynum LANGUAGES CXX)
# Explicitly set the C++ compiler
set(CMAKE_CXX_COMPILER /usr/bin/g++)
# Rest of your CMakeLists.txt remains the same
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MODULE_NAME dtypes)
set(SRC_FILE pynum/src/dtypes.cpp)
cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..
pybind11_add_module(${MODULE_NAME} ${SRC_FILE})
target_include_directories(${MODULE_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${MODULE_NAME} PRIVATE ${pybind11_INCLUDE_DIRS})
