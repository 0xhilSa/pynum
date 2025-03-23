<p align="center">
  <img src="./docs/pynum_light.png" alt="pynum">
</p>
<p align="center"><strong><em>a small python library for 1D and 2D arrays with GPU support</em></strong></p>

## Accelerators
- CUDA

## Prerequisites
  - GCC Compiler
  - NVCC Compiler [NVIDIA TOOLKIT](https://developer.nvidia.com/cuda-downloads)

## Installation
  ```bash
  git clone "git@github.com:0xhilSa/pynum.git"
  cd pynum && rm -rf .git*
  bash install.sh -y
  ```

## ToDo
- [ ] Complete the `__setitem__` method
- [ ] Work on the `astype` method
- [ ] Implement arithmetic and logical operations for `Vector` class

## Future Enhancements
- Implement an LLVM-based accelerator for CPU execution

## LICENSE
[MIT](./LICENSE)
