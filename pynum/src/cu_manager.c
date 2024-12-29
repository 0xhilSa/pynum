#include <Python.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                      \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(err)); \
            return NULL;                                     \
        }                                                    \
    } while (0)

// Allocate memory on the CUDA device
static PyObject* py_cuda_alloc(PyObject* self, PyObject* args) {
    PyObject* py_host_vec;
    if (!PyArg_ParseTuple(args, "O", &py_host_vec)) {
        return NULL;
    }

    if (!PyList_Check(py_host_vec)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of integers");
        return NULL;
    }

    size_t length = PyList_Size(py_host_vec);
    void* device_ptr = NULL;
    size_t size = length * sizeof(int);
    CUDA_CHECK(cudaMalloc(&device_ptr, size));

    return PyLong_FromVoidPtr(device_ptr);
}

// Copy memory from host to device
static PyObject* py_memcpy_htod(PyObject* self, PyObject* args) {
  PyObject* py_host_vec;
  PyObject* py_device_ptr;

  if(!PyArg_ParseTuple(args, "OO", &py_device_ptr, &py_host_vec)){ return NULL; }

  if(!PyList_Check(py_host_vec)){
    PyErr_SetString(PyExc_TypeError, "Expected a list of integers");
    return NULL;
  }

  int* device_ptr = (int*)PyLong_AsVoidPtr(py_device_ptr);
  size_t length = PyList_Size(py_host_vec);
  int* host_vec = (int*)malloc(length * sizeof(int));

  if(!host_vec){
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
      return NULL;
  }

  for(size_t i = 0; i < length; i++){ host_vec[i] = (int)PyLong_AsLong(PyList_GetItem(py_host_vec, i)); }

  size_t size = length * sizeof(int);
  CUDA_CHECK(cudaMemcpy(device_ptr, host_vec, size, cudaMemcpyHostToDevice));
  free(host_vec);

  Py_RETURN_NONE;
}

// Copy memory from device to host
static PyObject* py_memcpy_dtoh(PyObject* self, PyObject* args) {
  PyObject* py_device_ptr;
  Py_ssize_t length;

  if(!PyArg_ParseTuple(args, "On", &py_device_ptr, &length)){ return NULL; }

  int* device_ptr = (int*)PyLong_AsVoidPtr(py_device_ptr);
  int* host_vec = (int*)malloc(length * sizeof(int));

  if(!host_vec){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }

  size_t size = length * sizeof(int);
  CUDA_CHECK(cudaMemcpy(host_vec, device_ptr, size, cudaMemcpyDeviceToHost));

  PyObject* py_host_vec = PyList_New(length);
  for(Py_ssize_t i = 0; i < length; i++){
    PyList_SetItem(py_host_vec, i, PyLong_FromLong(host_vec[i]));
  }

  free(host_vec);
  return py_host_vec;
}

// Free CUDA device memory
static PyObject* py_cuda_free(PyObject* self, PyObject* args){
  PyObject* py_device_ptr;
  if(!PyArg_ParseTuple(args, "O", &py_device_ptr)){ return NULL; }
  int* device_ptr = (int*)PyLong_AsVoidPtr(py_device_ptr);
  if(device_ptr){ CUDA_CHECK(cudaFree(device_ptr)); }
  Py_RETURN_NONE;
}

// Query available CUDA device memory
static PyObject* py_cuda_query_free_memory(PyObject* self, PyObject* args){
  size_t free_mem = 0, total_mem = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  return PyLong_FromSize_t(free_mem);
}

// Method definitions
static PyMethodDef CuManagerMethods[] = {
  {"cuda_alloc", py_cuda_alloc, METH_VARARGS, "Allocate memory on the CUDA device"},
  {"memcpy_htod", py_memcpy_htod, METH_VARARGS, "Copy int vector from host to CUDA device"},
  {"memcpy_dtoh", py_memcpy_dtoh, METH_VARARGS, "Copy int vector from CUDA device to host"},
  {"cuda_free", py_cuda_free, METH_VARARGS, "Free CUDA device memory"},
  {"cuda_query_free_memory", py_cuda_query_free_memory, METH_NOARGS, "Query available CUDA device memory"},
  {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cu_manager_module = {
  PyModuleDef_HEAD_INIT,
  "cu_manager",
  "CUDA memory management module",
  -1,
  CuManagerMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_cu_manager(void){
  return PyModule_Create(&cu_manager_module);
}







