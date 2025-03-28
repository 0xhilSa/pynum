<p align="center">
  <img src="./docs/pynum_light.png" alt="pynum">
</p>
<p align="center"><strong><em>a small python library for 1D and 2D arrays with GPU support</em></strong></p>

## 🚀 Accelerators
- CUDA

## 📌 Prerequisites
  - GCC Compiler
  - NVCC Compiler [NVIDIA TOOLKIT](https://developer.nvidia.com/cuda-downloads)

## ⚡ Installation
  ```bash
  git clone "git@github.com:0xhilSa/pynum.git" ~/pynum
  cd pynum && rm -rf .git*
  bash install.sh -y
  ```

## ✅ ToDo
- [X] Implement the `__setitem__` method
- [ ] Develop the `astype` method
- [X] To solve (core dumped) issue while adding 2 vectors of dtype complex128
- [ ] Add arithmetic and logical operations for the `Vector` class
- [ ] Ensure compatibility with Windows OS

## 🔥 Future Enhancements
- Introduction to Matrix dtype
- Implement an LLVM-based accelerator for CPU execution

> [!IMPORTANT]
> Arithmetic ops on vectors can only be performed if they have the same data type
> - (short - short) ✔️
> - (short - int) ❌

## 📜 LICENSE
[MIT](./LICENSE)
