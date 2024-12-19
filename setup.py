from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
  Pybind11Extension(
    "pynum.dtypes",  # Module name
    ["pynum/src/dtypes.cpp"],  # Source file
  ),
]

pkg_name = {
  "pynum": ["dtypes.pyi"],
}

setup(
  name = "pynum",
  version = "0.0.1",
  packages = find_packages(),
  ext_modules = ext_modules,
  cmdclass = {"build_ext": build_ext},
  package_data = pkg_name,
  author = "Sahil Rajwar"
)

