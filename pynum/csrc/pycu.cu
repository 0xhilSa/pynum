#include <driver_types.h>
#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <python3.10/floatobject.h>
#include <python3.10/modsupport.h>
#include <python3.10/object.h>
#include <python3.10/pyerrors.h>
#include <python3.10/pyport.h>

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

static PyObject* setitem_index(PyObject* self, PyObject* args){
  PyObject *capsule, *value;
  Py_ssize_t length, index;
  const char* fmt;

  if(!PyArg_ParseTuple(args, "OOnns", &capsule, &value, &length, &index, &fmt)) return NULL;

  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_memory");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }

  size_t size = get_type_size(fmt);
  if(size == 0) return NULL;

  if(index < 0 || index >= length){
    PyErr_SetString(PyExc_IndexError, "Index out of range");
    return NULL;
  }

  void* host_value = malloc(size);
  if(!host_value){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate the host memory");
    return NULL;
  }

  if(strcmp(fmt, "?") == 0)      *(bool*)host_value = PyObject_IsTrue(value);
  else if(strcmp(fmt, "b") == 0) *(char*)host_value = (char)PyLong_AsLong(value);
  else if(strcmp(fmt, "B") == 0) *(unsigned char*)host_value = (unsigned char)PyLong_AsUnsignedLong(value);
  else if(strcmp(fmt, "h") == 0) *(short*)host_value = (short)PyLong_AsLong(value);
  else if(strcmp(fmt, "H") == 0) *(unsigned short*)host_value = (unsigned short)PyLong_AsUnsignedLong(value);
  else if(strcmp(fmt, "i") == 0) *(int*)host_value = (int)PyLong_AsLong(value);
  else if(strcmp(fmt, "I") == 0) *(unsigned int*)host_value = (unsigned int)PyLong_AsUnsignedLong(value);
  else if(strcmp(fmt, "l") == 0) *(long*)host_value = PyLong_AsLong(value);
  else if(strcmp(fmt, "L") == 0) *(unsigned long*)host_value = PyLong_AsUnsignedLong(value);
  else if(strcmp(fmt, "q") == 0) *(long long*)host_value = PyLong_AsLongLong(value);
  else if(strcmp(fmt, "Q") == 0) *(unsigned long long*)host_value = PyLong_AsUnsignedLongLong(value);
  else if(strcmp(fmt, "f") == 0) *(float*)host_value = (float)PyFloat_AsDouble(value);
  else if(strcmp(fmt, "d") == 0) *(double*)host_value = PyFloat_AsDouble(value);
  else if(strcmp(fmt, "g") == 0) *(long double*)host_value = (long double)PyFloat_AsDouble(value);
  else if(strcmp(fmt, "F") == 0){
    Py_complex pyc = PyComplex_AsCComplex(value);
    ((cuFloatComplex*)host_value)->x = (float)pyc.real;
    ((cuFloatComplex*)host_value)->y = (float)pyc.imag;
  }
  else if(strcmp(fmt, "D") == 0){
    Py_complex pyc = PyComplex_AsCComplex(value);
    ((cuDoubleComplex*)host_value)->x = pyc.real;
    ((cuDoubleComplex*)host_value)->y = pyc.imag;
  }
  else {
    free(host_value);
    PyErr_SetString(PyExc_TypeError, "Unsupported data type for assignment");
    return NULL;
  }

  CUDA_ERROR(cudaMemcpy((char*)device_ptr + index * size, host_value, size, cudaMemcpyHostToDevice));
  free(host_value);

  Py_RETURN_NONE;
}

static PyObject* setitem_slice(PyObject* self, PyObject* args){
  PyObject *capsule, *values;
  Py_ssize_t length, start, stop, step;
  const char* fmt;

  if(!PyArg_ParseTuple(args, "OOnnnns", &capsule, &values, &length, &start, &stop, &step, &fmt)) return NULL;

  void* device_ptr = PyCapsule_GetPointer(capsule, "cuda_memory");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }

  size_t size = get_type_size(fmt);
  if(size == 0) return NULL;

  if(start < 0 || stop <= start || step <= 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid slice indices");
    return NULL;
  }

  Py_ssize_t num_elements = (stop - start + step - 1) / step;

  if(!PySequence_Check(values) || PySequence_Length(values) != num_elements){
    PyErr_SetString(PyExc_ValueError, "Slice values must match slice length");
    return NULL;
  }

  void* host_values = malloc(num_elements * size);
  if(!host_values){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  for(Py_ssize_t i = 0; i < num_elements; i++){
    PyObject* value = PySequence_GetItem(values, i);
    void* elem_ptr = (char*)host_values + i * size;

    if(strcmp(fmt, "?") == 0)      *(bool*)elem_ptr = PyObject_IsTrue(value);
    else if(strcmp(fmt, "b") == 0) *(char*)elem_ptr = (char)PyLong_AsLong(value);
    else if(strcmp(fmt, "B") == 0) *(unsigned char*)elem_ptr = (unsigned char)PyLong_AsUnsignedLong(value);
    else if(strcmp(fmt, "h") == 0) *(short*)elem_ptr = (short)PyLong_AsLong(value);
    else if(strcmp(fmt, "H") == 0) *(unsigned short*)elem_ptr = (unsigned short)PyLong_AsUnsignedLong(value);
    else if(strcmp(fmt, "i") == 0) *(int*)elem_ptr = (int)PyLong_AsLong(value);
    else if(strcmp(fmt, "I") == 0) *(unsigned int*)elem_ptr = (unsigned int)PyLong_AsUnsignedLong(value);
    else if(strcmp(fmt, "l") == 0) *(long*)elem_ptr = PyLong_AsLong(value);
    else if(strcmp(fmt, "L") == 0) *(unsigned long*)elem_ptr = PyLong_AsUnsignedLong(value);
    else if(strcmp(fmt, "q") == 0) *(long long*)elem_ptr = PyLong_AsLongLong(value);
    else if(strcmp(fmt, "Q") == 0) *(unsigned long long*)elem_ptr = PyLong_AsUnsignedLongLong(value);
    else if(strcmp(fmt, "f") == 0) *(float*)elem_ptr = (float)PyFloat_AsDouble(value);
    else if(strcmp(fmt, "d") == 0) *(double*)elem_ptr = PyFloat_AsDouble(value);
    else if(strcmp(fmt, "g") == 0) *(long double*)elem_ptr = (long double)PyFloat_AsDouble(value);
    else if(strcmp(fmt, "F") == 0){
      Py_complex pyc = PyComplex_AsCComplex(value);
      ((cuFloatComplex*)elem_ptr)->x = (float)pyc.real;
      ((cuFloatComplex*)elem_ptr)->y = (float)pyc.imag;
    }
    else if(strcmp(fmt, "D") == 0){
      Py_complex pyc = PyComplex_AsCComplex(value);
      ((cuDoubleComplex*)elem_ptr)->x = pyc.real;
      ((cuDoubleComplex*)elem_ptr)->y = pyc.imag;
    }
    else {
      free(host_values);
      PyErr_SetString(PyExc_TypeError, "Unsupported data type for slice assignment");
      return NULL;
    }

    Py_DECREF(value);
  }

  for(Py_ssize_t i = 0; i < num_elements; i++){
    Py_ssize_t device_index = start + i * step;
    CUDA_ERROR(cudaMemcpy(
      (char*)device_ptr + device_index * size, 
      (char*)host_values + i * size, 
      size, 
      cudaMemcpyHostToDevice
    ));
  }

  free(host_values);
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
  {"toCuda", toCuda, METH_VARARGS, "Move host memory to CUDA device memory"},
  {"toHost", toHost, METH_VARARGS, "Move host memory to Host memory"},
  {"getitem_index", getitem_index, METH_VARARGS, "Get value via index"},
  {"getitem_slice", getitem_slice, METH_VARARGS, "Get value via slice"},
  {"setitem_index", setitem_index, METH_VARARGS, "set item through index"},
  {"setitem_slice", setitem_slice, METH_VARARGS, "set item through slice"},
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
