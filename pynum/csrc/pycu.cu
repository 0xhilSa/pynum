#include <Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define CUDA_CHECK(call)                                        \
  do{                                                           \
    cudaError_t err =  (call);                                  \
    if(err != cudaSuccess){                                     \
      PyErr_Format(PyExc_RuntimeError,                          \
      "CUDA Error: %s (FILE %s, LINE %d)",                      \
      cudaGetErrorString(err), __FILE__, __LINE__);             \
      return NULL;                                              \
    }                                                           \
  }while(0)


void free_capsule(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, "array_pointer");
  if (ptr) free(ptr);
}

static void cuda_free(PyObject* capsule){
  if(!PyCapsule_CheckExact(capsule)) {
    fprintf(stderr, "cuda_free: Expected a PyCapsule\n");
    return;
  }
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){
    fprintf(stderr, "cuda_free: Invalid PyCapsule (NULL pointer)\n");
    return;
  }cudaFree(device_ptr);
}

static PyObject* toCuda(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule");
    return NULL;
  }
  void* host_ptr = PyCapsule_GetPointer(capsule, "array_pointer");
  if(!host_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  void* device_ptr = NULL;
  size_t element_size;
  if(strcmp(fmt, "b") == 0) element_size = length * sizeof(char);
  else if(strcmp(fmt, "B") == 0) element_size = length * sizeof(unsigned char);
  else if(strcmp(fmt, "h") == 0) element_size = length * sizeof(short);
  else if(strcmp(fmt, "H") == 0) element_size = length * sizeof(unsigned short);
  else if(strcmp(fmt, "i") == 0) element_size = length * sizeof(int);
  else if(strcmp(fmt, "I") == 0) element_size = length * sizeof(unsigned int);
  else if(strcmp(fmt, "l") == 0) element_size = length * sizeof(long);
  else if(strcmp(fmt, "L") == 0) element_size = length * sizeof(unsigned long);
  else if(strcmp(fmt, "q") == 0) element_size = length * sizeof(long long);
  else if(strcmp(fmt, "Q") == 0) element_size = length * sizeof(unsigned long long);
  else if(strcmp(fmt, "f") == 0) element_size = length * sizeof(float);
  else if(strcmp(fmt, "d") == 0) element_size = length * sizeof(double);
  else if(strcmp(fmt, "g") == 0) element_size = length * sizeof(long double);
  else if(strcmp(fmt, "F") == 0) element_size = length * sizeof(cuFloatComplex);
  else if(strcmp(fmt, "D") == 0) element_size = length * sizeof(cuDoubleComplex);
  else if(strcmp(fmt, "G") == 0){
    PyErr_SetString(PyExc_NotImplementedError, "'long double complex' dtype not supported for now!!");
    return NULL;
  }else if(strcmp(fmt, "?") == 0) element_size = length * sizeof(bool);
  else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  CUDA_CHECK(cudaMalloc(&device_ptr, element_size));
  CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, element_size, cudaMemcpyHostToDevice));
  return PyCapsule_New(device_ptr, "cuda_array_pointer", cuda_free);
}

static PyObject* toHost(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  size_t element_size;
  if(strcmp(fmt, "b") == 0) element_size = length * sizeof(char);
  else if(strcmp(fmt, "B") == 0) element_size = length * sizeof(unsigned char);
  else if(strcmp(fmt, "h") == 0) element_size = length * sizeof(short);
  else if(strcmp(fmt, "H") == 0) element_size = length * sizeof(unsigned short);
  else if(strcmp(fmt, "i") == 0) element_size = length * sizeof(int);
  else if(strcmp(fmt, "I") == 0) element_size = length * sizeof(unsigned int);
  else if(strcmp(fmt, "l") == 0) element_size = length * sizeof(long);
  else if(strcmp(fmt, "L") == 0) element_size = length * sizeof(unsigned long);
  else if(strcmp(fmt, "q") == 0) element_size = length * sizeof(long long);
  else if(strcmp(fmt, "Q") == 0) element_size = length * sizeof(unsigned long long);
  else if(strcmp(fmt, "f") == 0) element_size = length * sizeof(float);
  else if(strcmp(fmt, "d") == 0) element_size = length * sizeof(double);
  else if(strcmp(fmt, "g") == 0) element_size = length * sizeof(long double);
  else if(strcmp(fmt, "F") == 0) element_size = length * sizeof(cuFloatComplex);
  else if(strcmp(fmt, "D") == 0) element_size = length * sizeof(cuDoubleComplex);
  else if(strcmp(fmt, "?") == 0) element_size = length * sizeof(bool);
  else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  void* host_ptr = malloc(length * element_size);
  if(!host_ptr){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, element_size, cudaMemcpyDeviceToHost));
  return PyCapsule_New(host_ptr, "array_pointer", free_capsule);
}

// that func isn't working as expected :(
static PyObject* py_get_value(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t index;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &index, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){ 
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  size_t element_size;
  if(strcmp(fmt, "b") == 0) element_size = sizeof(char);
  else if(strcmp(fmt, "B") == 0) element_size = sizeof(unsigned char);
  else if(strcmp(fmt, "h") == 0) element_size = sizeof(short);
  else if(strcmp(fmt, "H") == 0) element_size = sizeof(unsigned short);
  else if(strcmp(fmt, "i") == 0) element_size = sizeof(int);
  else if(strcmp(fmt, "I") == 0) element_size = sizeof(unsigned int);
  else if(strcmp(fmt, "l") == 0) element_size = sizeof(long);
  else if(strcmp(fmt, "L") == 0) element_size = sizeof(unsigned long);
  else if(strcmp(fmt, "q") == 0) element_size = sizeof(long long);
  else if(strcmp(fmt, "Q") == 0) element_size = sizeof(unsigned long long);
  else if(strcmp(fmt, "f") == 0) element_size = sizeof(float);
  else if(strcmp(fmt, "d") == 0) element_size = sizeof(double);
  else if(strcmp(fmt, "?") == 0) element_size = sizeof(bool);
  else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  void* host_value = malloc(element_size);
  if(!host_value){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }
  cudaMemcpy(host_value, (char*)device_ptr + index * element_size, element_size, cudaMemcpyDeviceToHost);
  PyObject* result = NULL;
  if(strcmp(fmt, "b") == 0) result = PyLong_FromLong(*(char*)host_value);
  else if(strcmp(fmt, "B") == 0) result = PyLong_FromUnsignedLong(*(unsigned char*)host_value);
  else if(strcmp(fmt, "h") == 0) result = PyLong_FromLong(*(short*)host_value);
  else if(strcmp(fmt, "H") == 0) result = PyLong_FromUnsignedLong(*(unsigned short*)host_value);
  else if(strcmp(fmt, "i") == 0) result = PyLong_FromLong(*(int*)host_value);
  else if(strcmp(fmt, "I") == 0) result = PyLong_FromUnsignedLong(*(unsigned int*)host_value);
  else if(strcmp(fmt, "l") == 0) result = PyLong_FromLong(*(long*)host_value);
  else if(strcmp(fmt, "L") == 0) result = PyLong_FromUnsignedLong(*(unsigned long*)host_value);
  else if(strcmp(fmt, "q") == 0) result = PyLong_FromLongLong(*(long long*)host_value);
  else if(strcmp(fmt, "Q") == 0) result = PyLong_FromUnsignedLongLong(*(unsigned long long*)host_value);
  else if(strcmp(fmt, "f") == 0) result = PyFloat_FromDouble(*(float*)host_value);
  else if(strcmp(fmt, "d") == 0) result = PyFloat_FromDouble(*(double*)host_value);
  else if(strcmp(fmt, "?") == 0) result = PyBool_FromLong(*(bool*)host_value);
  free(host_value);
  return result;
}

static PyMethodDef Methods[] = {
  {"toCuda", toCuda, METH_VARARGS, "copy memory from host to device"},
  {"toHost", toHost, METH_VARARGS, "copy memory from device to host"},
  {"get", py_get_value, METH_VARARGS, "get the value at a specified index"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pycu_module = {
  PyModuleDef_HEAD_INIT,
  "pycu",
  "CUDA device support for python",
  -1,
  Methods
};

PyMODINIT_FUNC PyInit_pycu(void){ return PyModule_Create(&pycu_module); }