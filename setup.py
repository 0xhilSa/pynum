from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

python_include_dir = os.path.join(sys.prefix, "include", f"python{sys.version_info.major}.{sys.version_info.minor}")

ext_modules = [
  Pybind11Extension(
    "pynum.dtypes",
    ["pynum/src/dtypes.cpp"],
    include_dirs=[python_include_dir],
    extra_compile_args = ["-std=c++17"],
    language = "c++"
  ),
]

pkg_name = {
  "pynum": ["dtypes.pyi"],
}

setup(
  name = "pynum",
  version = "0.0.1",
  url = "https://github.com/0xhilSa/pynum",
  packages = find_packages(),
  ext_modules = ext_modules,
  cmdclass = {"build_ext": build_ext},
  package_data = pkg_name,
  author = "Sahil Rajwar",
  long_description_content_type = "text/markdown",
  python_requires = ">=3.10",
  description = "a small python library for 1D and 2D arrays with GPU supports (WIP)"
)
