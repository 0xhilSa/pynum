#include <driver_types.h>
#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

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

//static PyObject* getitem(PyObject* )

static PyMethodDef methods[] = {
  {"toCuda", toCuda, METH_VARARGS, "Move host memory to CUDA device memory"},
  {"toHost", toHost, METH_VARARGS, "Move host memory to Host memory"},
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
