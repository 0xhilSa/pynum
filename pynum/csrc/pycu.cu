#include <driver_types.h>
#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <python3.10/object.h>
#include <python3.10/pyerrors.h>

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


__global__ void strided_copy_kernel(char* dst, const char* src, size_t size, Py_ssize_t start, Py_ssize_t step, Py_ssize_t num_elements){
  Py_ssize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_elements){
    size_t src_index = start + idx * step;
    memcpy(dst + idx * size, src + src_index * size, size);
  }
}


static size_t get_type_size(const char* fmt){
  switch(*fmt){
    case '?': return sizeof(bool);
    case 'b': return sizeof(char);
    case 'B': return sizeof(unsigned char);
    case 'h': return sizeof(short);
    case 'H': return sizeof(unsigned short);
    case 'i': return sizeof(int);
    case 'I': return sizeof(unsigned int);
    case 'l': return sizeof(long);
    case 'L': return sizeof(unsigned long);
    case 'q': return sizeof(long long);
    case 'Q': return sizeof(unsigned long long);
    case 'f': return sizeof(float);
    case 'd': return sizeof(double);
    case 'g': return sizeof(long double);
    case 'F': return sizeof(cuFloatComplex);
    case 'D': return sizeof(cuDoubleComplex);
    case 'G': {
                PyErr_SetString(PyExc_RuntimeError, "DType: 'long double complex' isn't supported");
                return 0;
              }
    default: return 0;
  }
}

static void cuda_free(PyObject* capsule){
  void* ptr = PyCapsule_GetPointer(capsule, "cuda_memory");
  if(ptr) cudaFree(ptr);
}

static void free_memory(PyObject* capsule){
  void* ptr = PyCapsule_GetPointer(capsule, "host_memory");
  if(ptr) free(ptr);
}

static PyObject* toCuda(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &length, &fmt)) return NULL;
  void* host_ptr = PyCapsule_GetPointer(capsule, "host_memory");
  if(!host_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid host data capsule");
    return NULL;
  }
  size_t type_size = get_type_size(fmt);
  if(type_size == 0) return NULL;
  void* device_ptr;
  CUDA_ERROR(cudaMalloc(&device_ptr, length * type_size));
  CUDA_ERROR(cudaMemcpy(device_ptr, host_ptr, length * type_size, cudaMemcpyHostToDevice));
  return PyCapsule_New(device_ptr, "cuda_memory", cuda_free);
}

static PyObject* toHost(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &length, &fmt)) return NULL;
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_memory");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA memory capsule");
    return NULL;
  }
  size_t type_size = get_type_size(fmt);
  if(type_size == 0) return NULL;
  void* host_ptr = malloc(length * type_size);
  if(!host_ptr){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(host_ptr, device_ptr, length * type_size, cudaMemcpyDeviceToHost));
  return PyCapsule_New(host_ptr, "host_memory", free_memory);
}

static PyObject* getitem_index(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t index;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &index, &fmt)) return NULL;

  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_memory");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA memory capsule");
    return NULL;
  }

  size_t size = get_type_size(fmt);
  if(size == 0){ return NULL; }

  void* res;
  CUDA_ERROR(cudaMalloc(&res, size));
  CUDA_ERROR(cudaMemcpy(res, (char*)device_ptr + index * size, size, cudaMemcpyDeviceToDevice));
  return PyCapsule_New(res, "cuda_memory", cuda_free);
}

static PyObject* getitem_slice(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t start, stop, step;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Onnns", &capsule, &start, &stop, &step, &fmt)) return NULL;
  
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_memory");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA memory capsule");
    return NULL;
  }

  size_t size = get_type_size(fmt);
  if(size == 0) return NULL;

  if(start < 0 || stop <= start || step <= 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid slice indices");
    return NULL;
  }
  Py_ssize_t num_elements = (stop - start + step - 1) / step;
  size_t slice_size = num_elements * size;

  void* slice_ptr;
  CUDA_ERROR(cudaMalloc(&slice_ptr, slice_size));

  int threadsPerBlock = 256;
  int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  strided_copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    (char*)slice_ptr, (char*)device_ptr, size, start, step, num_elements
  );

  return PyCapsule_New(slice_ptr, "cuda_memory", cuda_free);
}

static PyMethodDef methods[] = {
  {"toCuda", toCuda, METH_VARARGS, "Move host memory to CUDA device memory"},
  {"toHost", toHost, METH_VARARGS, "Move host memory to Host memory"},
  {"getitem_index", getitem_index, METH_VARARGS, "Get value via index"},
  {"getitem_slice", getitem_slice, METH_VARARGS, "Get value via slice"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "pycu",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_pycu(void){
  return PyModule_Create(&module);
}
