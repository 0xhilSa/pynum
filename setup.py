from setuptools import setup, find_packages

setup(
  name="pynum",
  version="0.0.1",
  author="Sahil Rajwar",
  license="MIT",
  description="a small python library for 1D and 2D arrays with GPU support",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  packages=find_packages(),
  package_data={"pynum.csrc": ["*.so", "*.pyi"]},
  classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.6",
)


