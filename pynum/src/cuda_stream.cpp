#include <Python.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <string>


#define CUDA_CHECK(call)                                                        \
  do{                                                                           \
    cudaError_t err = call;                                                     \
    if(err != cudaSuccess){                                                     \
      PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err));             \
      return NULL;                                                              \
    }                                                                           \
  }while (0)


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

static PyObject* py_cuda_alloc_short(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(short)); }
static PyObject* py_cuda_alloc_int(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(int)); }
static PyObject* py_cuda_alloc_long(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(long)); }
static PyObject* py_cuda_alloc_float(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(float)); }
static PyObject* py_cuda_alloc_double(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(double)); }
static PyObject* py_cuda_alloc_bool(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(bool)); }
static PyObject* py_cuda_alloc_complex(PyObject* self, PyObject* args){ return py_cuda_alloc_generic(self, args, sizeof(cuDoubleComplex)); }


static PyObject* py_memcpy_htod_generic(PyObject* self, PyObject* args, size_t element_size, PyObject* (*converter)(PyObject*)){
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
    else if(element_size == sizeof(cuDoubleComplex)){
      PyObject* real = PyObject_GetAttrString(item, "real");
      PyObject* imag = PyObject_GetAttrString(item, "imag");
      if(!real || !imag || !PyNumber_Check(real) || !PyNumber_Check(imag)){
        Py_XDECREF(real);
        Py_XDECREF(imag);
        free(host_vec);
        PyErr_SetString(PyExc_TypeError, "List items must have real and imaginary components as numbers");
        return NULL;
      }
      double real_val = PyFloat_AsDouble(real);
      double imag_val = PyFloat_AsDouble(imag);
      Py_DECREF(real);
      Py_DECREF(imag);
      if(PyErr_Occurred()){
        free(host_vec);
        return NULL;
      }
      ((cuDoubleComplex*)host_vec)[i] = make_cuDoubleComplex(PyFloat_AsDouble(real), PyFloat_AsDouble(imag));
    }
    else if(element_size == sizeof(bool)) ((bool*)host_vec)[i] = PyObject_IsTrue(converted);
    if(converted != item) Py_DECREF(converted);
  }
  CUDA_CHECK(cudaMemcpy(device_ptr, host_vec, length * element_size, cudaMemcpyHostToDevice));
  free(host_vec);
  Py_RETURN_NONE;
}

static PyObject* py_memcpy_htod_short(PyObject* self, PyObject* args){ return py_memcpy_htod_generic(self, args, sizeof(short), PyNumber_Long); }
static PyObject* py_memcpy_htod_int(PyObject* self, PyObject* args){  return py_memcpy_htod_generic(self, args, sizeof(int), PyNumber_Long); }
static PyObject* py_memcpy_htod_long(PyObject* self, PyObject* args){ return py_memcpy_htod_generic(self, args, sizeof(long), PyNumber_Long); }
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

static PyObject* py_memcpy_htod_complex(PyObject* self, PyObject* args) {
  PyObject* py_device_ptr;
  PyObject* py_host_vec;

  if (!PyArg_ParseTuple(args, "OO", &py_device_ptr, &py_host_vec)) return NULL;

  if (!PyList_Check(py_host_vec)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list");
    return NULL;
  }

  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  size_t length = PyList_Size(py_host_vec);
  cuDoubleComplex* host_vec = (cuDoubleComplex*)malloc(length * sizeof(cuDoubleComplex));

  if (!host_vec) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  for (size_t i = 0; i < length; i++) {
    PyObject* item = PyList_GetItem(py_host_vec, i);
    if (!PyComplex_Check(item)) {
      free(host_vec);
      PyErr_SetString(PyExc_TypeError, "List items must be complex numbers");
      return NULL;
    }
    double real = PyComplex_RealAsDouble(item);
    double imag = PyComplex_ImagAsDouble(item);
    host_vec[i] = make_cuDoubleComplex(real, imag);
  }

  CUDA_CHECK(cudaMemcpy(device_ptr, host_vec, length * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  free(host_vec);

  Py_RETURN_NONE;
}

static PyObject* py_memcpy_htod_bool(PyObject* self, PyObject* args){
  return py_memcpy_htod_generic(self, args, sizeof(bool), [](PyObject* item) -> PyObject* {
    if(!item){
      PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
      return NULL;
    }
    return PyBool_FromLong(PyObject_IsTrue(item));
  });
}

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
    else if(element_size == sizeof(cuDoubleComplex)){
      cuDoubleComplex value = ((cuDoubleComplex*)host_vec)[i];
      item = PyComplex_FromDoubles(cuCreal(value), cuCimag(value));
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

static PyObject* py_memcpy_dtoh_short(PyObject* self, PyObject* args){ return py_memcpy_dtoh_generic(self, args, sizeof(short), NULL); }
static PyObject* py_memcpy_dtoh_int(PyObject* self, PyObject* args){ return py_memcpy_dtoh_generic(self, args, sizeof(int), NULL); }
static PyObject* py_memcpy_dtoh_long(PyObject* self, PyObject* args){ return py_memcpy_dtoh_generic(self, args, sizeof(long), NULL); }
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
  CUDA_CHECK(cudaMemcpy(host_vec, device_ptr, length * sizeof(double), cudaMemcpyDeviceToHost));
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
static PyObject* py_memcpy_dtoh_complex(PyObject* self, PyObject* args){ return py_memcpy_dtoh_generic(self, args, sizeof(cuDoubleComplex), NULL); }
static PyObject* py_memcpy_dtoh_bool(PyObject* self, PyObject* args){ return py_memcpy_dtoh_generic(self, args, sizeof(bool), NULL); }


static PyObject* py_cuda_free(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  if(!PyArg_ParseTuple(args, "O", &py_device_ptr)) return NULL;
  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr) CUDA_CHECK(cudaFree(device_ptr));
  Py_RETURN_NONE;
}

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

static PyObject* py_get_value_complex(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  Py_ssize_t index;
  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &index)) return NULL;
  void* device_ptr = PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  cuDoubleComplex host_value;
  CUDA_CHECK(cudaMemcpy(&host_value, (char*)device_ptr + index * sizeof(cuDoubleComplex), sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  return PyComplex_FromDoubles(cuCreal(host_value), cuCimag(host_value));
}

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
    cuDoubleComplex value;
    CUDA_CHECK(cudaMemcpy(&value, (char*)device_ptr + current * sizeof(cuDoubleComplex), sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    PyObject* py_value = PyComplex_FromDoubles(cuCreal(value), cuCimag(value));
    PyList_SetItem(py_list, i, py_value);
  }
  return py_list;
}

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
  cuDoubleComplex value = make_cuDoubleComplex(real, imag);
  cuDoubleComplex* device_ptr = (cuDoubleComplex*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid device pointer");
    return NULL;
  }
  CUDA_CHECK(cudaMemcpy(device_ptr + index, &value, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  Py_RETURN_NONE;
}

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

static PyObject* py_cuda_query_free_memory(PyObject* self, PyObject* args){
  size_t free_mem = 0, total_mem = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  return PyLong_FromSize_t(free_mem);
}

static PyObject* py_count_device(PyObject* self, PyObject* args){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    PyErr_SetString(PyExc_RuntimeError, "no CUDA-capable devices detected");
    return NULL;
  }
  return PyLong_FromLong(count);
}

static PyObject* py_cuda_select_device(PyObject* self, PyObject* args){
  int device_index;
  if(!PyArg_ParseTuple(args, "i", &device_index)){
    PyErr_SetString(PyExc_ValueError, "Expected an integer device index.");
    return NULL;
  }
  CUDA_CHECK(cudaSetDevice(device_index));
  Py_RETURN_NONE;
}

static PyMethodDef CuManagerMethods[] = {
  {"py_alloc_short", py_cuda_alloc_short, METH_VARARGS, "allocate memory on CUDA device for short dtype"},
  {"py_alloc_int", py_cuda_alloc_int, METH_VARARGS, "allocate memory on CUDA device for int dtype"},
  {"py_alloc_long", py_cuda_alloc_long, METH_VARARGS, "allocate memory on CUDA device for long dtype"},
  {"py_alloc_float", py_cuda_alloc_float, METH_VARARGS, "allocate memory on CUDA device for float dtype"},
  {"py_alloc_double", py_cuda_alloc_double, METH_VARARGS, "allocate memory on CUDA device for double dtype"},
  {"py_alloc_bool", py_cuda_alloc_bool, METH_VARARGS, "allocate memory on CUDA device for bool dtype"},
  {"py_alloc_complex", py_cuda_alloc_complex, METH_VARARGS, "allocate memory on CUDA device for short dtype"},
  {"py_memcpy_htod_short", py_memcpy_htod_short, METH_VARARGS, "copy memory from host to device for short dtype"},
  {"py_memcpy_htod_int", py_memcpy_htod_int, METH_VARARGS, "copy memory from host to device for int dtype"},
  {"py_memcpy_htod_long", py_memcpy_htod_long, METH_VARARGS, "copy memory from host to device for long dtype"},
  {"py_memcpy_htod_float", py_memcpy_htod_float, METH_VARARGS, "copy memory from host to device for float dtype"},
  {"py_memcpy_htod_double", py_memcpy_htod_double, METH_VARARGS, "copy memory from host to device for double dtype"},
  {"py_memcpy_htod_complex", py_memcpy_htod_complex, METH_VARARGS, "copy memory from host to device for complex dtype"},
  {"py_memcpy_htod_bool", py_memcpy_htod_bool, METH_VARARGS, "copy memory from host to device for bool dtype"},
  {"py_memcpy_dtoh_short", py_memcpy_dtoh_short, METH_VARARGS, "copy memory from device to host for short dtype"},
  {"py_memcpy_dtoh_int", py_memcpy_dtoh_int, METH_VARARGS, "copy memory from device to host for int dtype"},
  {"py_memcpy_dtoh_long", py_memcpy_dtoh_long, METH_VARARGS, "copy memory from device to host for long dtype"},
  {"py_memcpy_dtoh_float", py_memcpy_dtoh_float, METH_VARARGS, "copy memory from device to host for float dtype"},
  {"py_memcpy_dtoh_double", py_memcpy_dtoh_double, METH_VARARGS, "copy memory from device to host for double dtype"},
  {"py_memcpy_dtoh_complex", py_memcpy_dtoh_complex, METH_VARARGS, "copy memory from device to host for complex dtype"},
  {"py_memcpy_dtoh_bool", py_memcpy_dtoh_bool, METH_VARARGS, "copy memory from device to host for bool dtype"},
  {"py_get_value_short", py_get_value_short, METH_VARARGS, "get a short value from the device at a specific index"},
  {"py_get_value_int", py_get_value_int, METH_VARARGS, "get a int value from the device at a specific index"},
  {"py_get_value_long", py_get_value_long, METH_VARARGS, "get a long value from the device at a specific index"},
  {"py_get_value_float", py_get_value_float, METH_VARARGS, "get a float value from the device at a specific index"},
  {"py_get_value_double", py_get_value_double, METH_VARARGS, "get a double value from the device at a specific index"},
  {"py_get_value_complex", py_get_value_complex, METH_VARARGS, "get a complex value from the device at a specific index"},
  {"py_get_value_bool", py_get_value_bool, METH_VARARGS, "get a bool value from the device at a specific index"},
  {"py_get_slice_short", py_get_slice_short, METH_VARARGS, "get short sliced value from the device within the range"},
  {"py_get_slice_int", py_get_slice_int, METH_VARARGS, "get int sliced value from the device within the range"},
  {"py_get_slice_long", py_get_slice_long, METH_VARARGS, "get long sliced value from the device within the range"},
  {"py_get_slice_float", py_get_slice_float, METH_VARARGS, "get float sliced value from the device within the range"},
  {"py_get_slice_double", py_get_slice_double, METH_VARARGS, "get double sliced value from the device within the range"},
  {"py_get_slice_complex", py_get_slice_complex, METH_VARARGS, "get complex sliced value from the device within the range"},
  {"py_get_slice_bool", py_get_slice_bool, METH_VARARGS, "get bool sliced value from the device within the range"},
  {"py_set_value_short", py_set_value_short, METH_VARARGS, "set short value from the device to a specific index"},
  {"py_set_value_int", py_set_value_int, METH_VARARGS, "set int value from the device to a specific index"},
  {"py_set_value_long", py_set_value_long, METH_VARARGS, "set long value from the device to a specific index"},
  {"py_set_value_float", py_set_value_float, METH_VARARGS, "set float value from the device to a specific index"},
  {"py_set_value_double", py_set_value_double, METH_VARARGS, "set double value from the device to a specific index"},
  {"py_set_value_complex", py_set_value_complex, METH_VARARGS, "set complex value from the device to a specific index"},
  {"py_set_value_bool", py_set_value_bool, METH_VARARGS, "set bool value from the device to a specific index"},
  {"py_count_device", py_count_device, METH_VARARGS, "returns the number of CUDA device available"},
  {"py_cuda_select_device", py_cuda_select_device, METH_VARARGS, "select the CUDA device through device index"},
  {"py_free", py_cuda_free, METH_VARARGS, "free the pointer"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cuda_stream_module = {
  PyModuleDef_HEAD_INIT,
  "cuda_stream",
  "Extended CUDA mempry management module",
  -1,
  CuManagerMethods
};

PyMODINIT_FUNC PyInit_cuda_stream(void){ return PyModule_Create(&cuda_stream_module); }
