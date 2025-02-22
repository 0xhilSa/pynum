#include <python3.10/Python.h>
#include <cuda_runtime.h>

#define CUDA_ERROR(call)                                        \
  do{                                                           \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess){                                    \
      PyErr_Format(PyExc_RuntimeError,                          \
          "CUDA_ERROR: %s at %s:%d",                            \
          cudaGetErrorString(err), __FILE__, __LINE__);         \
      return nullptr;                                           \
    }                                                           \
  }while (0)



