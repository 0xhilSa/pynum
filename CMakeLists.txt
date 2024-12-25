cmake_minimum_required(VERSION 3.15)

# Enable verbose output during compilation
set(CMAKE_VERBOSE_MAKEFILE ON)

project(pynum LANGUAGES CXX)

# Find Python first
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")

# Find pybind11
find_package(pybind11 REQUIRED)
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define source files
set(SOURCE_FILES pynum/src/dtypes.cpp)

# Create the module
pybind11_add_module(dtypes ${SOURCE_FILES})

# Set output directory
set_target_properties(dtypes PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pynum
)

# Add explicit include directories
target_include_directories(dtypes PRIVATE 
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Link against Python libraries
target_link_libraries(dtypes PRIVATE Python::Python)
