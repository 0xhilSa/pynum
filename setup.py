from setuptools import setup, find_packages

setup(
  name = "pynum",
  version = "0.0.1",
  url = "https://github.com/0xhilSa/pynum",
  license = "MIT",
  packages = find_packages(include=["pynum", "pynum.src"]),
  package_data= {"pynum.src": ["*.so", "*.pyi"]},
  author = "Sahil Rajwar",
  author_email = "UNKNOWN",
  long_description_content_type = "text/markdown",
  python_requires = ">=3.10",
  description = "a small python library for 1D and 2D arrays with GPU supports (WIP)"
)
