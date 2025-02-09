#include <Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

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


__global__ void strided_copy_kernel(void* dst, void* src, size_t start, size_t step, size_t length, size_t element_size){
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < length){
    char* src_base = (char*)src;
    char* dst_base = (char*)dst;
    memcpy(dst_base + index * element_size, src_base + (start + index * step) * element_size, element_size);
  }
}

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

static PyObject* py_get_by_value(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length, index;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Onns", &capsule, &length, &index, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule");
    return NULL;
  }
  if(index < 0) index += length;
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){ 
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  size_t element_size;
  if(strcmp(fmt, "b") == 0 || strcmp(fmt, "B") == 0) element_size = sizeof(char);
  else if(strcmp(fmt, "h") == 0 || strcmp(fmt, "H") == 0) element_size = sizeof(short);
  else if(strcmp(fmt, "i") == 0 || strcmp(fmt, "I") == 0) element_size = sizeof(int);
  else if(strcmp(fmt, "l") == 0 || strcmp(fmt, "L") == 0) element_size = sizeof(long);
  else if(strcmp(fmt, "q") == 0 || strcmp(fmt, "Q") == 0) element_size = sizeof(long long);
  else if(strcmp(fmt, "f") == 0) element_size = sizeof(float);
  else if(strcmp(fmt, "d") == 0) element_size = sizeof(double);
  else if(strcmp(fmt, "F") == 0) element_size = sizeof(cuFloatComplex);
  else if(strcmp(fmt, "D") == 0) element_size = sizeof(cuDoubleComplex);
  else if(strcmp(fmt, "?") == 0) element_size = sizeof(bool);
  else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  void* device_result_ptr = NULL;
  CUDA_CHECK(cudaMalloc(&device_result_ptr, element_size));
  CUDA_CHECK(cudaMemcpy(device_result_ptr, (char*)device_ptr + index * element_size, element_size, cudaMemcpyDeviceToDevice));
  return PyCapsule_New(device_result_ptr, "cuda_array_pointer", cuda_free);
}

static PyObject* py_get_by_slice(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length, start, stop, step;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Onnnns", &capsule, &length, &start, &stop, &step, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected PyCapsule");
    return NULL;
  }
  if(start < 0) start += length;
  if(stop < 0) stop += length;
  if(start < 0 || stop < start || step == 0){
    PyErr_SetString(PyExc_ValueError, "Invalid slice indices");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  size_t element_size;
  if(strcmp(fmt, "b") == 0 || strcmp(fmt, "B") == 0) element_size = sizeof(char);
  else if(strcmp(fmt, "h") == 0 || strcmp(fmt, "H") == 0) element_size = sizeof(short);
  else if(strcmp(fmt, "i") == 0 || strcmp(fmt, "I") == 0) element_size = sizeof(int);
  else if(strcmp(fmt, "l") == 0 || strcmp(fmt, "L") == 0) element_size = sizeof(long);
  else if(strcmp(fmt, "q") == 0 || strcmp(fmt, "Q") == 0) element_size = sizeof(long long);
  else if(strcmp(fmt, "f") == 0) element_size = sizeof(float);
  else if(strcmp(fmt, "d") == 0) element_size = sizeof(double);
  else if(strcmp(fmt, "F") == 0) element_size = sizeof(cuFloatComplex);
  else if(strcmp(fmt, "D") == 0) element_size = sizeof(cuDoubleComplex);
  else if(strcmp(fmt, "?") == 0) element_size = sizeof(bool);
  else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  size_t sliced_length = (stop - start) / step;
  void* device_result_ptr = NULL;
  CUDA_CHECK(cudaMalloc(&device_result_ptr, sliced_length * element_size));
  if (step == 1) CUDA_CHECK(cudaMemcpy(device_result_ptr, (char*)device_ptr + start * element_size, sliced_length * element_size, cudaMemcpyDeviceToDevice));
  else{
    strided_copy_kernel<<<(sliced_length + 255) / 256, 256>>>(device_result_ptr, device_ptr, start, step, sliced_length, element_size);
    CUDA_CHECK(cudaGetLastError());
  }
  return PyCapsule_New(device_result_ptr, "cuda_array_pointer", cuda_free);
}

static PyObject* py_set_by_index(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length, index;
  const char* fmt;
  PyObject* value;
  if(!PyArg_ParseTuple(args, "OnnsO", &capsule, &length, &index, &fmt, &value)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected PyCapsule");
    return NULL;
  }
  if(index < 0) index += length; 
  if(index < 0 || index >= length){
    PyErr_SetString(PyExc_IndexError, "Index out of range");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  size_t element_size;
  void* temp = NULL;
  if(strcmp(fmt, "b") == 0){
    element_size = sizeof(char);
    temp = malloc(element_size);
    *(char*)temp = (char)PyLong_AsLong(value);
  }else if(strcmp(fmt, "B") == 0){
    element_size = sizeof(unsigned char);
    temp = malloc(element_size);
    *(unsigned char*)temp = (unsigned char)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "h") == 0){
    element_size = sizeof(short);
    temp = malloc(element_size);
    *(short*)temp = (short)PyLong_AsLong(value);
  }else if(strcmp(fmt, "H") == 0){
    element_size = sizeof(unsigned short);
    temp = malloc(element_size);
    *(unsigned short*)temp = (unsigned short)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "i") == 0){
    element_size = sizeof(int);
    temp = malloc(element_size);
    *(int*)temp = (int)PyLong_AsLong(value);
  }else if(strcmp(fmt, "I") == 0){
    element_size = sizeof(unsigned int);
    temp = malloc(element_size);
    *(unsigned int*)temp = (unsigned int)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "l") == 0){
    element_size = sizeof(long);
    temp = malloc(element_size);
    *(long*)temp = (long)PyLong_AsLong(value);
  }else if(strcmp(fmt, "L") == 0){
    element_size = sizeof(unsigned long);
    temp = malloc(element_size);
    *(unsigned long*)temp = (unsigned long)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "q") == 0){
    element_size = sizeof(long long);
    temp = malloc(element_size);
    *(long long*)temp = (long long)PyLong_AsLongLong(value);
  }else if(strcmp(fmt, "Q") == 0){
    element_size = sizeof(unsigned long long);
    temp = malloc(element_size);
    *(unsigned long long*)temp = (unsigned long long)PyLong_AsUnsignedLongLong(value);
  }else if(strcmp(fmt, "f") == 0){
    element_size = sizeof(float);
    temp = malloc(element_size);
    *(float*)temp = (float)PyFloat_AsDouble(value);
  }else if(strcmp(fmt, "d") == 0){
    element_size = sizeof(double);
    temp = malloc(element_size);
    *(double*)temp = (double)PyFloat_AsDouble(value);
  }else if(strcmp(fmt, "g") == 0){
    element_size = sizeof(long double);
    temp = malloc(element_size);
    *(long double*)temp = (long double)PyFloat_AsDouble(value);
  }else if(strcmp(fmt, "F") == 0){
    element_size = sizeof(cuFloatComplex);
    temp = malloc(element_size);
    float real = (float)PyComplex_RealAsDouble(value);
    float imag = (float)PyComplex_ImagAsDouble(value);
    *(cuFloatComplex*)temp = make_cuFloatComplex(real, imag);
  }else if(strcmp(fmt, "D") == 0){
    element_size = sizeof(cuDoubleComplex);
    temp = malloc(element_size);
    double real = (double)PyComplex_RealAsDouble(value);
    double imag = (double)PyComplex_ImagAsDouble(value);
    *(cuDoubleComplex*)temp = make_cuDoubleComplex(real, imag);
  }else if(strcmp(fmt, "?") == 0){
    element_size = sizeof(bool);
    temp = malloc(element_size);
    *(bool*)temp = (bool)PyObject_IsTrue(value);
  }else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  cudaMemcpy((char*)device_ptr + index * element_size, temp, element_size, cudaMemcpyHostToDevice);
  free(temp);
  Py_RETURN_NONE;
}

static PyObject* py_set_by_slice(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length, start, stop, step;
  const char* fmt;
  PyObject* value;
  if(!PyArg_ParseTuple(args, "OnnnnsO", &capsule, &length, &start, &stop, &step, &fmt, &value)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected PyCapsule");
    return NULL;
  }
  if(start < 0) start += length;
  if(stop < 0) stop += length;
  if(start < 0 || stop > length || step <= 0 || start >= stop){
    PyErr_SetString(PyExc_IndexError, "Invalid slice range");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_array_pointer");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  size_t element_size;
  void* temp = NULL;
  if(strcmp(fmt, "b") == 0){
    element_size = sizeof(char);
    temp = malloc(element_size);
    *(char*)temp = (char)PyLong_AsLong(value);
  }else if(strcmp(fmt, "B") == 0){
    element_size = sizeof(unsigned char);
    temp = malloc(element_size);
    *(unsigned char*)temp = (unsigned char)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "h") == 0){
    element_size = sizeof(short);
    temp = malloc(element_size);
    *(short*)temp = (short)PyLong_AsLong(value);
  }else if(strcmp(fmt, "H") == 0){
    element_size = sizeof(unsigned short);
    temp = malloc(element_size);
    *(unsigned short*)temp = (unsigned short)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "i") == 0){
    element_size = sizeof(int);
    temp = malloc(element_size);
    *(int*)temp = (int)PyLong_AsLong(value);
  }else if(strcmp(fmt, "I") == 0){
    element_size = sizeof(unsigned int);
    temp = malloc(element_size);
    *(unsigned int*)temp = (unsigned int)PyLong_AsUnsignedLong(value);
  }else if(strcmp(fmt, "f") == 0){
    element_size = sizeof(float);
    temp = malloc(element_size);
    *(float*)temp = (float)PyFloat_AsDouble(value);
  }else if(strcmp(fmt, "d") == 0){
    element_size = sizeof(double);
    temp = malloc(element_size);
    *(double*)temp = (double)PyFloat_AsDouble(value);
  }else if(strcmp(fmt, "F") == 0){
    element_size = sizeof(cuFloatComplex);
    temp = malloc(element_size);
    float real = (float)PyComplex_RealAsDouble(value);
    float imag = (float)PyComplex_ImagAsDouble(value);
    *(cuFloatComplex*)temp = make_cuFloatComplex(real, imag);
  }else if(strcmp(fmt, "D") == 0){
    element_size = sizeof(cuDoubleComplex);
    temp = malloc(element_size);
    double real = (double)PyComplex_RealAsDouble(value);
    double imag = (double)PyComplex_ImagAsDouble(value);
    *(cuDoubleComplex*)temp = make_cuDoubleComplex(real, imag);
  }else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  for(Py_ssize_t i = start; i < stop; i += step){
    cudaMemcpy((char*)device_ptr + i * element_size, temp, element_size, cudaMemcpyHostToDevice);
  }
  free(temp);
  Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
  {"toCuda", toCuda, METH_VARARGS, "copy memory from host to device"},
  {"toHost", toHost, METH_VARARGS, "copy memory from device to host"},
  {"get_by_index", py_get_by_value, METH_VARARGS, "get the value at a specified index"},
  {"get_by_slice", py_get_by_slice, METH_VARARGS, "get the values at a specified range"},
  {"set_by_index", py_set_by_index, METH_VARARGS, "set value by index"},
  {"set_by_slice", py_set_by_slice, METH_VARARGS, "set value by sluce"},
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
