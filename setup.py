from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

module = Extension(
  "pynum",
  sources = ["pynum/src/cuda_stream.cpp"],
  libraries = ["cuda","cudart"]
)


setup(
  name = "pynum",
  version = "0.0.1",
  url = "https://github.com/0xhilSa/pynum",
  license = "MIT",
  packages = find_packages(include=["pynum", "pynum.src"]),
  ext_modules = [module],
  package_data= {"pynum.src": ["*.so"]},
  author = "Sahil Rajwar",
  long_description_content_type = "text/markdown",
  python_requires = ">=3.10",
  description = "a small python library for 1D and 2D arrays with GPU supports (WIP)"
)
