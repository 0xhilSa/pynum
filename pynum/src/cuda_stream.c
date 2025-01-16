#include <Python.h>
#include <cuda_runtime.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <stdbool.h>

#define CUDA_CHECK(call) \
  do{ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
      PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err)); \
      return NULL; \
    } \
  }while (0)



// generic allocation function
static PyObject* py_cuda_alloc_generic(PyObject* self, PyObject* args, size_t element_size){
  PyObject* py_host_vec;
  if(!PyArg_ParseTuple(args, "O", &py_host_vec)) return NULL;

  if(!PyList_Check(py_host_vec)){
    PyErr_SetString(PyExc_TypeError, "Expected a list");
    return NULL;
  }

  size_t length = PyList_Size(py_host_vec);
  void* device_ptr = NULL;
  size_t size = length * element_size;
  CUDA_CHECK(cudaMalloc(&device_ptr, size));

  return PyLong_FromVoidPtr(device_ptr);
}

// type-specific allocation functions
static PyObject* py_cuda_alloc_short(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(short));
}

static PyObject* py_cuda_alloc_int(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(int));
}

static PyObject* py_cuda_alloc_long(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(long));
}

static PyObject* py_cuda_alloc_float(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(float));
}

static PyObject* py_cuda_alloc_double(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(double));
}

static PyObject* py_cuda_alloc_complex(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(double complex));
}

static PyObject* py_cuda_alloc_bool(PyObject* self, PyObject* args){
  return py_cuda_alloc_generic(self, args, sizeof(bool));
}

// generic host to device copy function
static PyObject* py_memcpy_htod_generic(PyObject* self, PyObject* args, size_t element_size, PyObject* (*converter)(PyObject*)) {
  PyObject* py_device_ptr;
  PyObject* py_host_vec;

  if(!PyArg_ParseTuple(args, "OO", &py_device_ptr, &py_host_vec)) return NULL;

  if(!PyList_Check(py_host_vec)){
    PyErr_SetString(PyExc_TypeError, "Expected a list");
    return NULL;
  }

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  size_t length = PyList_Size(py_host_vec);
  void* host_vec = malloc(length * element_size);

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  // type-specific conversion
  for(size_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(py_host_vec, i);
    PyObject* converted = converter(item);
    if(!converted){
      free(host_vec);
      return NULL;
    }

    if(element_size == sizeof(short)) ((short*)host_vec)[i] = (short)PyLong_AsLong(converted);
    else if(element_size == sizeof(int)) ((int*)host_vec)[i] = (int)PyLong_AsLong(converted);
    else if(element_size == sizeof(long)) ((long*)host_vec)[i] = PyLong_AsLong(converted);
    else if(element_size == sizeof(float)) ((float*)host_vec)[i] = PyFloat_AsDouble(converted);
    else if(element_size == sizeof(double)) ((double*)host_vec)[i] = PyFloat_AsDouble(converted);
    else if(element_size == sizeof(double complex)){
      PyObject* real = PyObject_GetAttrString(item, "real");
      PyObject* imag = PyObject_GetAttrString(item, "imag");
      ((double complex*)host_vec)[i] = PyFloat_AsDouble(real) + PyFloat_AsDouble(imag) * I;
      Py_DECREF(real);
      Py_DECREF(imag);
    }
    else if(element_size == sizeof(bool)) ((bool*)host_vec)[i] = PyObject_IsTrue(converted) ? true : false;
    
    if(converted != item) Py_DECREF(converted);
  }

  CUDA_CHECK(cudaMemcpy(device_ptr, host_vec, length * element_size, cudaMemcpyHostToDevice));
  free(host_vec);

  Py_RETURN_NONE;
}

// type-specific host to device copy functions
static PyObject* py_memcpy_htod_short(PyObject* self, PyObject* args){
  return py_memcpy_htod_generic(self, args, sizeof(short), PyNumber_Long);
}

static PyObject* py_memcpy_htod_int(PyObject* self, PyObject* args){
  return py_memcpy_htod_generic(self, args, sizeof(int), PyNumber_Long);
}

static PyObject* py_memcpy_htod_long(PyObject* self, PyObject* args){
  return py_memcpy_htod_generic(self, args, sizeof(long), PyNumber_Long);
}

static PyObject* py_memcpy_htod_float(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  PyObject* py_host_vec;

  if(!PyArg_ParseTuple(args, "OO", &py_device_ptr, &py_host_vec)) return NULL;
  if(!PyList_Check(py_host_vec)){
    PyErr_SetString(PyExc_TypeError, "Expected a list of floats");
    return NULL;
  }

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  size_t length = PyList_Size(py_host_vec);
  float* host_vec = (float*)malloc(length * sizeof(float));

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  for(size_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(py_host_vec, i);
    if(!PyNumber_Check(item)){
      free(host_vec);
      PyErr_SetString(PyExc_TypeError, "List contains non-numeric types");
      return NULL;
    }
    host_vec[i] = PyFloat_AsDouble(item);
    if(PyErr_Occurred()){
      free(host_vec);
      return NULL;
    }
  }

  cudaError_t err = cudaMemcpy(device_ptr, host_vec, length * sizeof(float), cudaMemcpyHostToDevice);
  free(host_vec);

  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* py_memcpy_htod_double(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  PyObject* py_host_vec;

  if(!PyArg_ParseTuple(args, "OO", &py_device_ptr, &py_host_vec)) return NULL;
  if(!PyList_Check(py_host_vec)){
    PyErr_SetString(PyExc_TypeError, "Expected a list of floats");
    return NULL;
  }

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  size_t length = PyList_Size(py_host_vec);
  double* host_vec = (double*)malloc(length * sizeof(double));

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  for(size_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(py_host_vec, i);
    if(!PyNumber_Check(item)){
      free(host_vec);
      PyErr_SetString(PyExc_TypeError, "List contains non-numeric types");
      return NULL;
      }
    host_vec[i] = PyFloat_AsDouble(item);
    if(PyErr_Occurred()){
      free(host_vec);
      return NULL;
    }
  }

  cudaError_t err = cudaMemcpy(device_ptr, host_vec, length * sizeof(double), cudaMemcpyHostToDevice);
  free(host_vec);

  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* py_memcpy_htod_complex(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  PyObject* py_host_vec;

  if(!PyArg_ParseTuple(args, "OO", &py_device_ptr, &py_host_vec)) return NULL;

  if(!PyList_Check(py_host_vec)){
    PyErr_SetString(PyExc_TypeError, "Expected a list");
    return NULL;
  }

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  size_t length = PyList_Size(py_host_vec);
  double complex* host_vec = malloc(length * sizeof(double complex));

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  for(size_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(py_host_vec, i);
    if (!PyComplex_Check(item)) {
      free(host_vec);
      PyErr_SetString(PyExc_TypeError, "List items must be complex numbers");
      return NULL;
    }
    double real = PyComplex_RealAsDouble(item);
    double imag = PyComplex_ImagAsDouble(item);
    host_vec[i] = real + imag * I;
  }

  CUDA_CHECK(cudaMemcpy(device_ptr, host_vec, length * sizeof(double complex), cudaMemcpyHostToDevice));
  free(host_vec);

  Py_RETURN_NONE;
}

static PyObject* convert_to_bool(PyObject* obj){
  int is_true = PyObject_IsTrue(obj);
  if(is_true == -1){
    PyErr_SetString(PyExc_ValueError, "Invalid boolean value");
    return NULL;
  }
  return PyBool_FromLong(is_true);
}

static PyObject* py_memcpy_htod_bool(PyObject* self, PyObject* args){
  return py_memcpy_htod_generic(self, args, sizeof(bool), convert_to_bool);
}

static PyObject* py_memcpy_htoh_generic(PyObject* self, PyObject* args, size_t element_size) {
  PyObject* py_src_vec;
  PyObject* py_dst_vec;

  if(!PyArg_ParseTuple(args, "OO", &py_src_vec, &py_dst_vec)) return NULL;
  if(!PyList_Check(py_src_vec) || !PyList_Check(py_dst_vec)){
    PyErr_SetString(PyExc_TypeError, "Both arguments must be lists");
    return NULL;
  }

  size_t src_length = PyList_Size(py_src_vec);
  size_t dst_length = PyList_Size(py_dst_vec);

  if(src_length != dst_length){
    PyErr_SetString(PyExc_ValueError, "Source and destination lists must have the same length");
    return NULL;
  }

  void* src_data = malloc(src_length * element_size);
  void* dst_data = malloc(dst_length * element_size);

  if(!src_data || !dst_data){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for source or destination vectors");
    return NULL;
  }

  for(size_t i = 0; i < src_length; i++){
    PyObject* item = PyList_GetItem(py_src_vec, i);
    if(element_size == sizeof(short)) ((short*)src_data)[i] = (short)PyLong_AsLong(item);
    else if(element_size == sizeof(int)) ((int*)src_data)[i] = (int)PyLong_AsLong(item);
    else if(element_size == sizeof(long)) ((long*)src_data)[i] = PyLong_AsLong(item);
    else if(element_size == sizeof(float)) ((float*)src_data)[i] = (float)PyFloat_AsDouble(item);
    else if(element_size == sizeof(double)) ((double*)src_data)[i] = PyFloat_AsDouble(item);
    else if(element_size == sizeof(double complex)){
      PyObject* real = PyObject_GetAttrString(item, "real");
      PyObject* imag = PyObject_GetAttrString(item, "imag");
      ((double complex*)src_data)[i] = PyFloat_AsDouble(real) + PyFloat_AsDouble(imag) * I;
      Py_DECREF(real);
      Py_DECREF(imag);
    }
    else if(element_size == sizeof(bool)) ((bool*)src_data)[i] = PyObject_IsTrue(item) ? true : false;
  }

  memcpy(dst_data, src_data, src_length * element_size);
  for(size_t i = 0; i < dst_length; i++){
    PyObject* item = NULL;
    if(element_size == sizeof(short)) item = PyLong_FromLong(((short*)dst_data)[i]);
    else if(element_size == sizeof(int)) item = PyLong_FromLong(((int*)dst_data)[i]);
    else if(element_size == sizeof(long)) item = PyLong_FromLong(((long*)dst_data)[i]);
    else if(element_size == sizeof(float)) item = PyFloat_FromDouble(((float*)dst_data)[i]);
    else if(element_size == sizeof(double)) item = PyFloat_FromDouble(((double*)dst_data)[i]);
    else if (element_size == sizeof(double complex)){
      double complex value = ((double complex*)dst_data)[i];
      item = PyComplex_FromDoubles(creal(value), cimag(value));
    }
    else if(element_size == sizeof(bool)) item = PyBool_FromLong(((bool*)dst_data)[i] ? 1 : 0);

    if(!item){
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert value to Python object");
      free(src_data);
      free(dst_data);
      return NULL;
    }

    PyList_SetItem(py_dst_vec, i, item);
  }

  free(src_data);
  free(dst_data);

  Py_RETURN_NONE;
}

// Type-specific htoh functions
static PyObject* py_memcpy_htoh_short(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(short));
}

static PyObject* py_memcpy_htoh_int(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(int));
}

static PyObject* py_memcpy_htoh_long(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(long));
}

static PyObject* py_memcpy_htoh_float(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(float));
}

static PyObject* py_memcpy_htoh_double(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(double));
}

static PyObject* py_memcpy_htoh_complex(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(double complex));
}

static PyObject* py_memcpy_htoh_bool(PyObject* self, PyObject* args){
  return py_memcpy_htoh_generic(self, args, sizeof(bool));
}

// generic device to host copy function
static PyObject* py_memcpy_dtoh_generic(PyObject* self, PyObject* args, size_t element_size, PyObject* (*converter)(void*)){
  PyObject* py_device_ptr;
  Py_ssize_t length;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &length)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  void* host_vec = malloc(length * element_size);

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  CUDA_CHECK(cudaMemcpy(host_vec, device_ptr, length * element_size, cudaMemcpyDeviceToHost));

  PyObject* py_host_vec = PyList_New(length);
  for(Py_ssize_t i = 0; i < length; i++) {
    PyObject* item;
    if(element_size == sizeof(short)) item = PyLong_FromLong(((short*)host_vec)[i]);
    else if(element_size == sizeof(int)) item = PyLong_FromLong(((int*)host_vec)[i]);
    else if(element_size == sizeof(long))  item = PyLong_FromLong(((long*)host_vec)[i]);
    else if(element_size == sizeof(float))  item = PyFloat_FromDouble(((float*)host_vec)[i]);
    else if(element_size == sizeof(double)) item = PyFloat_FromDouble(((double*)host_vec)[i]);
    else if(element_size == sizeof(double complex)){
      double complex value = ((double complex*)host_vec)[i];
      item = PyComplex_FromDoubles(creal(value), cimag(value));
    }
    else if(element_size == sizeof(bool)) item = PyBool_FromLong(((bool*)host_vec)[i] ? 1 : 0);
    else item = NULL;

    if(!item){
      Py_DECREF(py_host_vec);
      free(host_vec);
      return NULL;
    }
    PyList_SET_ITEM(py_host_vec, i, item);
  }

  free(host_vec);
  return py_host_vec;
}

// type-specific device to host copy functions
static PyObject* py_memcpy_dtoh_short(PyObject* self, PyObject* args){
  return py_memcpy_dtoh_generic(self, args, sizeof(short), NULL);
}

static PyObject* py_memcpy_dtoh_int(PyObject* self, PyObject* args){
  return py_memcpy_dtoh_generic(self, args, sizeof(int), NULL);
}

static PyObject* py_memcpy_dtoh_long(PyObject* self, PyObject* args){
  return py_memcpy_dtoh_generic(self, args, sizeof(long), NULL);
}

static PyObject* py_memcpy_dtoh_float(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t length;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &length)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  float* host_vec = (float*)malloc(length * sizeof(float));

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  CUDA_CHECK(cudaMemcpy(host_vec, device_ptr, length * sizeof(float), cudaMemcpyDeviceToHost));
  PyObject* py_host_vec = PyList_New(length);
  if (!py_host_vec) {
    free(host_vec);
    return NULL;
  }

  for(Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PyFloat_FromDouble(host_vec[i]);
    if (!item) {
      Py_DECREF(py_host_vec);
      free(host_vec);
      return NULL;
    }
    PyList_SET_ITEM(py_host_vec, i, item);
  }

  free(host_vec);
  return py_host_vec;
}

static PyObject* py_memcpy_dtoh_double(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t length;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &length)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  double* host_vec = (double*)malloc(length * sizeof(double));

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(host_vec, device_ptr, length * sizeof(double), cudaMemcpyDeviceToHost));

  // create a new Python list to store the values
  PyObject* py_host_vec = PyList_New(length);
  if (!py_host_vec) {
    free(host_vec);
    return NULL;
  }

  // convert the data in the host vector into Python floats and add to the list
  for(Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PyFloat_FromDouble(host_vec[i]);
    if (!item) {
      Py_DECREF(py_host_vec);
      free(host_vec);
      return NULL;
    }
    PyList_SET_ITEM(py_host_vec, i, item);
  }

  free(host_vec);
  return py_host_vec;
}

static PyObject* py_memcpy_dtoh_complex(PyObject* self, PyObject* args){
  return py_memcpy_dtoh_generic(self, args, sizeof(double complex), NULL);
}

static PyObject* py_memcpy_dtoh_bool(PyObject* self, PyObject* args){
  return py_memcpy_dtoh_generic(self, args, sizeof(bool), NULL);
}

// device to device copy function remains largely unchanged but needs size parameter
static PyObject* py_memcpy_dtod_generic(PyObject* self, PyObject* args, size_t element_size) {
  PyObject* py_src_ptr;
  PyObject* py_dst_ptr;
  Py_ssize_t length;

  if(!PyArg_ParseTuple(args, "OOn", &py_dst_ptr, &py_src_ptr, &length)) return NULL;

  void* src_ptr = PyLong_AsVoidPtr(py_src_ptr);
  void* dst_ptr = PyLong_AsVoidPtr(py_dst_ptr);

  size_t size = length * element_size;
  CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice));
  Py_RETURN_NONE;
}

// type-specific device to device copy functions
static PyObject* py_memcpy_dtod_short(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(short));
}

static PyObject* py_memcpy_dtod_int(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(int));
}

static PyObject* py_memcpy_dtod_long(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(long));
}

static PyObject* py_memcpy_dtod_float(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(float));
}

static PyObject* py_memcpy_dtod_double(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(double));
}

static PyObject* py_memcpy_dtod_complex(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(double complex));
}

static PyObject* py_memcpy_dtod_bool(PyObject* self, PyObject* args){
  return py_memcpy_dtod_generic(self, args, sizeof(bool));
}

// CUDA free function remains unchanged
static PyObject* py_cuda_free(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  if(!PyArg_ParseTuple(args, "O", &py_device_ptr)) return NULL;
  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr) CUDA_CHECK(cudaFree(device_ptr));
  Py_RETURN_NONE;
}

// retrieve data from the list at a specific index for short
static PyObject* py_get_value_short(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }

  short host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(short), sizeof(short), cudaMemcpyDeviceToHost));
  return PyLong_FromLong(host_value);
}

// retrieve data from the list at a specific index for int
static PyObject* py_get_value_int(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }

  int host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(int), sizeof(int), cudaMemcpyDeviceToHost));
  return PyLong_FromLong(host_value);
}

// retrieve data from the list at a specific index for long
static PyObject* py_get_value_long(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  
  long host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(long), sizeof(long), cudaMemcpyDeviceToHost));
  return PyLong_FromLong(host_value); 
}

// retrieve data from the list at a specific index for float
static PyObject* py_get_value_float(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer!");
    return NULL;
  }

  float host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(float), sizeof(float), cudaMemcpyDeviceToHost));
  return PyFloat_FromDouble(host_value);
}

// retrieve data from the list at a specific index for double
static PyObject* py_get_value_double(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer!");
    return NULL;
  }

  double host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(double), sizeof(double), cudaMemcpyDeviceToHost));
  return PyFloat_FromDouble(host_value);
}

// retrieve data from the list at a specific index for complex
static PyObject* py_get_value_complex(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }

  double complex host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(double complex), sizeof(double complex), cudaMemcpyDeviceToHost));
  return PyComplex_FromDoubles(creal(host_value), cimag(host_value));
}

// retrieve data from the list at a specific index for bool
static PyObject* py_get_value_bool(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer!");
    return NULL;
  }

  bool host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(bool), sizeof(bool), cudaMemcpyDeviceToHost));
  return PyBool_FromLong(host_value);
}

// retrieve data from the list within the given range and steps for int
static PyObject* py_get_slice_short(PyObject* self, PyObject* args){
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;
  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;
  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    short value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(short), sizeof(short), cudaMemcpyDeviceToHost));
    PyList_SetItem(py_list, i, PyLong_FromLong(value));
  }
  return py_list;
}

// retrieve data from the list within the given range and steps for int
static PyObject* py_get_slice_int(PyObject* self, PyObject* args){
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;
  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;
  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    int value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(int), sizeof(int), cudaMemcpyDeviceToHost));
    PyList_SetItem(py_list, i, PyLong_FromLong(value));
  }
  return py_list;
}

// retrieve data from the list within the given range and steps for long
static PyObject* py_get_slice_long(PyObject* self, PyObject* args){
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;
  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;

  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    long value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(long), sizeof(long), cudaMemcpyDeviceToHost));
    PyList_SetItem(py_list, i, PyLong_FromLong(value));
  }
  return py_list;
}

// retrieve data from the list within the given range and steps for float
static PyObject* py_get_slice_float(PyObject* self, PyObject* args){
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;
  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;

  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    float value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(float), sizeof(float), cudaMemcpyDeviceToHost));
    PyList_SetItem(py_list, i, PyLong_FromLong(value));
  }
  return py_list;
}

// retrieve data from the list within the given range and steps for double
static PyObject* py_get_slice_double(PyObject* self, PyObject* args){
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;
  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;

  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    double value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(double), sizeof(double), cudaMemcpyDeviceToHost));
    PyList_SetItem(py_list, i, PyLong_FromLong(value));
  }
  return py_list;
}

// retrieve data from the list within the given raneg and steps for complex
static PyObject* py_get_slice_complex(PyObject* self, PyObject* args) {
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;

  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index.");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0.");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;

  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list.");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    double complex value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(double complex), sizeof(double complex), cudaMemcpyDeviceToHost));
    PyObject* py_value = PyComplex_FromDoubles(creal(value), cimag(value));
    PyList_SetItem(py_list, i, py_value);
  }
  return py_list;
}

// retrieve data from the list within the given range and steps for bool
static PyObject* py_get_slice_bool(PyObject* self, PyObject* args){
  void* device_ptr;
  size_t start, stop, steps;

  if(!PyArg_ParseTuple(args, "Kkkk", (unsigned long*)&device_ptr, &start, &stop, &steps)) return NULL;
  if(start >= stop){
    PyErr_SetString(PyExc_ValueError, "Start index must be less than stop index");
    return NULL;
  }

  if(steps == 0){
    PyErr_SetString(PyExc_ValueError, "Steps must be greater than 0");
    return NULL;
  }

  size_t num_elements = (stop - start + steps - 1) / steps;
  PyObject* py_list = PyList_New(num_elements);
  if(!py_list){
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
    return NULL;
  }

  for(size_t i = 0, current = start; i < num_elements; ++i, current += steps){
    bool value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(bool), sizeof(bool), cudaMemcpyDeviceToHost));
    PyList_SetItem(py_list, i, PyBool_FromLong(value));
  }
  return py_list;
}

// set a data from a list at a specific index for short
static PyObject* py_set_value_short(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  short value;

  if(!PyArg_ParseTuple(args, "Oni", &py_device_ptr, &index, &value)) return NULL;
  short* device_ptr = (short*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(short), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// set a data from a list at a specific index for int
static PyObject* py_set_value_int(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  int value;

  if(!PyArg_ParseTuple(args, "Oni", &py_device_ptr, &index, &value)) return NULL;
  int* device_ptr = (int*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(int), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// set a data from a list at a specific index for long
static PyObject* py_set_value_long(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  long value;

  if(!PyArg_ParseTuple(args, "Onl", &py_device_ptr, &index, &value)) return NULL;
  long* device_ptr = (long*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(long), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// set a data from a list at a specific index for float
static PyObject* py_set_value_float(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  float value;

  if(!PyArg_ParseTuple(args, "Ond", &py_device_ptr, &index, &value)) return NULL;
  float* device_ptr = (float*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(float), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// set a data from a list at a specific index for double
static PyObject* py_set_value_double(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  double value;

  if(!PyArg_ParseTuple(args, "Ond", &py_device_ptr, &index, &value)) return NULL;
  double* device_ptr = (double*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(double), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// set a data from a list at a specific index for complex
static PyObject* py_set_value_complex(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  PyObject* py_value;

  if(!PyArg_ParseTuple(args, "OnO", &py_device_ptr, &index, &py_value)) return NULL;
  if(!PyComplex_Check(py_value)){
    PyErr_SetString(PyExc_TypeError, "Value must be complex number");
    return NULL;
  }

  double real = PyComplex_RealAsDouble(py_value);
  double imag = PyComplex_ImagAsDouble(py_value);
  double complex value = real + imag * I;

  double complex* device_ptr = (double complex*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(double complex), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// set a data from a list at a specific index for bool
static PyObject* py_set_value_bool(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  bool value;

  if(!PyArg_ParseTuple(args, "Oni", &py_device_ptr, &index, &value)) return NULL;
  bool* device_ptr = (bool*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(bool), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

// memory query function remains unchanged
static PyObject* py_cuda_query_free_memory(PyObject* self, PyObject* args){
  size_t free_mem = 0, total_mem = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  return PyLong_FromSize_t(free_mem);
}

static PyObject* py_count_device(){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, "no CUDA-capable devices detected");
    return NULL;
  }
  return PyLong_FromLong(count);
}

static PyObject* py_cuda_available(){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess) Py_RETURN_FALSE;
  Py_RETURN_TRUE;
}

// method definitions for all data types
static PyMethodDef CuManagerMethods[] = {
  {"py_alloc_short", py_cuda_alloc_short, METH_VARARGS, "allocate memory on CUDA device for short data type"},
  {"py_alloc_int", py_cuda_alloc_int, METH_VARARGS, "allocate memory on CUDA device for integer data type"},
  {"py_alloc_long", py_cuda_alloc_long, METH_VARARGS, "allocate memory on CUDA device for long data type"},
  {"py_alloc_float", py_cuda_alloc_float, METH_VARARGS, "allocate memory on CUDA device for float data type"},
  {"py_alloc_double", py_cuda_alloc_double, METH_VARARGS, "allocate memory on CUDA device for double data type"},
  {"py_alloc_complex", py_cuda_alloc_complex, METH_VARARGS, "allocate memory on CUDA device for complex data type"},
  {"py_alloc_bool", py_cuda_alloc_bool, METH_VARARGS, "allocate memory on CUDA device for boolean data type"},
  {"py_memcpy_htod_short", py_memcpy_htod_short, METH_VARARGS, "copy memory from host to device for short data type"},
  {"py_memcpy_htod_int", py_memcpy_htod_int, METH_VARARGS, "copy memory from host to device for integer data type"},
  {"py_memcpy_htod_long", py_memcpy_htod_long, METH_VARARGS, "copy memory from host to device for long data type"},
  {"py_memcpy_htod_float", py_memcpy_htod_float, METH_VARARGS, "copy memory from host to device for float data type"},
  {"py_memcpy_htod_double", py_memcpy_htod_double, METH_VARARGS, "copy memory from host to device for double data type"},
  {"py_memcpy_htod_complex", py_memcpy_htod_complex, METH_VARARGS, "copy memory from host to device for complex data type"},
  {"py_memcpy_htod_bool", py_memcpy_htod_bool, METH_VARARGS, "copy memory from host to device for boolean data type"},
  {"py_memcpy_htoh_short", py_memcpy_htoh_short, METH_VARARGS, "copy memory from host to host for short data type"},
  {"py_memcpy_htoh_int", py_memcpy_htoh_int, METH_VARARGS, "copy memory from host to host for integer data type"},
  {"py_memcpy_htoh_long", py_memcpy_htoh_long, METH_VARARGS, "copy memory from host to host for long data type"},
  {"py_memcpy_htoh_float", py_memcpy_htoh_float, METH_VARARGS, "copy memory from host to host for float data type"},
  {"py_memcpy_htoh_double", py_memcpy_htoh_double, METH_VARARGS, "copy memory from host to host for double data type"},
  {"py_memcpy_htoh_complex", py_memcpy_htoh_complex, METH_VARARGS, "copy memory from host to host for complex data type"},
  {"py_memcpy_htoh_bool", py_memcpy_htoh_bool, METH_VARARGS, "copy memory from host to host for boolean data type"},
  {"py_memcpy_dtoh_short", py_memcpy_dtoh_short, METH_VARARGS, "copy memory from device to host for short data type"},
  {"py_memcpy_dtoh_int", py_memcpy_dtoh_int, METH_VARARGS, "copy memory from device to host for integer data type"},
  {"py_memcpy_dtoh_long", py_memcpy_dtoh_long, METH_VARARGS, "copy memory from device to host for long data type"},
  {"py_memcpy_dtoh_float", py_memcpy_dtoh_float, METH_VARARGS, "copy memory from device to host for float data type"},
  {"py_memcpy_dtoh_double", py_memcpy_dtoh_double, METH_VARARGS, "copy memory from device to host for double data type"},
  {"py_memcpy_dtoh_complex", py_memcpy_dtoh_complex, METH_VARARGS, "copy memory from device to host for complex data type"},
  {"py_memcpy_dtoh_bool", py_memcpy_dtoh_bool, METH_VARARGS, "copy memory from device to host for boolean data type"},
  {"py_memcpy_dtod_short", py_memcpy_dtod_short, METH_VARARGS, "copy memory from device to device for short data type"},
  {"py_memcpy_dtod_int", py_memcpy_dtod_int, METH_VARARGS, "copy memory from device to device for integer data type"},
  {"py_memcpy_dtod_long", py_memcpy_dtod_long, METH_VARARGS, "copy memory from device to device for long data type"},
  {"py_memcpy_dtod_float", py_memcpy_dtod_float, METH_VARARGS, "copy memory from device to device for float data type"},
  {"py_memcpy_dtod_double", py_memcpy_dtod_double, METH_VARARGS, "copy memory from device to device for double data type"},
  {"py_memcpy_dtod_complex", py_memcpy_dtod_complex, METH_VARARGS, "copy memory from device to device for complex data type"},
  {"py_memcpy_dtod_bool", py_memcpy_dtod_bool, METH_VARARGS, "copy memory from device to device for boolean data type"},
  {"py_free", py_cuda_free, METH_VARARGS, "free the memory from CUDA device"},
  {"py_query_free_memory", py_cuda_query_free_memory, METH_NOARGS, "query available CUDA device memory"},
  {"py_get_value_short", py_get_value_short, METH_VARARGS, "get an short value from the device at a specific index"},
  {"py_get_value_int", py_get_value_int, METH_VARARGS, "get an integer value from the device at a specific index"},
  {"py_get_value_long", py_get_value_long, METH_VARARGS, "get a long value from the device at a specific index"},
  {"py_get_value_float", py_get_value_float, METH_VARARGS, "get a float value from the device at a specific index"},
  {"py_get_value_double", py_get_value_double, METH_VARARGS, "get a double value from the device at a specific index"},
  {"py_get_value_complex", py_get_value_complex, METH_VARARGS, "get a complex value from the device at a specific index"},
  {"py_get_value_bool", py_get_value_bool, METH_VARARGS, "get a boolean value from the device at a specific index"},
  {"py_get_slice_short", py_get_slice_short, METH_VARARGS, "get short slice from device memory"},
  {"py_get_slice_int", py_get_slice_int, METH_VARARGS, "get integer slice from device memory"},
  {"py_get_slice_long", py_get_slice_long, METH_VARARGS, "get long slice from device memory"},
  {"py_get_slice_float", py_get_slice_float, METH_VARARGS, "get float slice from device memory"},
  {"py_get_slice_double", py_get_slice_double, METH_VARARGS, "get double slice from device memory"},
  {"py_get_slice_complex", py_get_slice_complex, METH_VARARGS, "get complex slice from device memory"},
  {"py_get_slice_bool", py_get_slice_bool, METH_VARARGS, "get boolean slice from device memory"},
  {"py_set_value_short", py_set_value_short, METH_VARARGS, "set an short value in device memory"},
  {"py_set_value_int", py_set_value_int, METH_VARARGS, "set an integer value in device memory"},
  {"py_set_value_long", py_set_value_long, METH_VARARGS, "set a long value in device memory"},
  {"py_set_value_float", py_set_value_float, METH_VARARGS, "set a float value in device memory"},
  {"py_set_value_double", py_set_value_double, METH_VARARGS, "set a double value in device memory"},
  {"py_set_value_complex", py_set_value_complex, METH_VARARGS, "set a complex value in device memory"},
  {"py_set_value_bool", py_set_value_bool, METH_VARARGS, "set a boool value in device memory"},
  {"py_count_device", py_count_device, METH_NOARGS, "returns the number of device available"},
  {"py_cuda_available", py_cuda_available, METH_NOARGS, "returns whether the CUDA device is available or not"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cuda_stream_module = {
  PyModuleDef_HEAD_INIT,
  "cuda_stream",
  "Extended CUDA memory management module",
  -1,
  CuManagerMethods
};

PyMODINIT_FUNC PyInit_cuda_stream(void){ return PyModule_Create(&cuda_stream_module); }
