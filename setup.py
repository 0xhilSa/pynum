from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

# Get Python include directory properly
python_include_dir = os.path.join(sys.prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')

ext_modules = [
  Pybind11Extension(
    "pynum.dtypes",  # Module name
    ["pynum/src/dtypes.cpp"],  # Source file
    include_dirs=[python_include_dir],  # Properly specify include directory
    extra_compile_args=['-std=c++17'],  # C++17 support
    language='c++'
  ),
]

pkg_name = {
  "pynum": ["dtypes.pyi"],
}

setup(
  name="pynum",
  version="0.0.1",
  packages=find_packages(),
  ext_modules=ext_modules,
  cmdclass={"build_ext": build_ext},
  package_data=pkg_name,
  author="Sahil Rajwar",
  python_requires='>=3.10',  # Added Python version requirement
)
