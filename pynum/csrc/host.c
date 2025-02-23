#include <python3.10/Python.h>
#include <stdlib.h>
#include <stdbool.h>
#include <complex.h>


static void free_memory(PyObject* capsule){
  void* ptr = PyCapsule_GetPointer(capsule, "host_memory");
  if(ptr) free(ptr);
}

static size_t get_type_size(const char *fmt){
  switch (*fmt) {
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
    case 'F': return sizeof(float complex);
    case 'D': return sizeof(double complex);
    case 'G': return sizeof(long double complex);
    default: return 0;
  }
}

// return the pycapsule using malloc
static PyObject* array(PyObject* self, PyObject* args){
  PyObject* py_array;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &py_array, &fmt)) return NULL;
  if(!PyList_Check(py_array)){
    PyErr_SetString(PyExc_TypeError, "Expected a python list");
    return NULL;
  }
  Py_ssize_t size = PyList_Size(py_array);
  size_t type_size = get_type_size(fmt);
  if(type_size == 0){
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
  void* buffer = malloc(size * type_size);
  if(!buffer){
    PyErr_NoMemory();
    return NULL;
  }
  for(Py_ssize_t i = 0; i < size; i++){
    PyObject* item = PyList_GetItem(py_array, i);
    if(!item){
      free(buffer);
      return NULL;
    }
    switch(*fmt){
      case '?': ((bool*)buffer)[i] = PyObject_IsTrue(item); break;
      case 'b': ((char*)buffer)[i] = (char)PyLong_AsLong(item); break;
      case 'B': ((unsigned char*)buffer)[i] = (unsigned char)PyLong_AsUnsignedLong(item); break;
      case 'h': ((short*)buffer)[i] = (short)PyLong_AsLong(item); break;
      case 'H': ((unsigned short*)buffer)[i] = (unsigned short)PyLong_AsUnsignedLong(item); break;
      case 'i': ((int*)buffer)[i] = (int)PyLong_AsLong(item); break;
      case 'I': ((unsigned int*)buffer)[i] = (unsigned int)PyLong_AsUnsignedLong(item); break;
      case 'l': ((long*)buffer)[i] = (long)PyLong_AsLong(item); break;
      case 'L': ((unsigned long*)buffer)[i] = (unsigned long)PyLong_AsUnsignedLong(item); break;
      case 'q': ((long long*)buffer)[i] = (long long)PyLong_AsLongLong(item); break;
      case 'Q': ((unsigned long long*)buffer)[i] = (unsigned long long)PyLong_AsUnsignedLongLong(item); break;
      case 'f': ((float*)buffer)[i] = (float)PyFloat_AsDouble(item); break;
      case 'd': ((double*)buffer)[i] = PyFloat_AsDouble(item); break;
      case 'g': ((long double*)buffer)[i] = PyFloat_AsDouble(item); break;
      case 'F': {
                  Py_complex cmpx = PyComplex_AsCComplex(item);
                  ((float complex*)buffer)[i] = (float)cmpx.real + (float)cmpx.imag * I;
                }break;
      case 'D': {
                  Py_complex cmpx = PyComplex_AsCComplex(item);
                  ((double complex*)buffer)[i] = (double)cmpx.real + (double)cmpx.imag * I;
                }break;
      case 'G': {
                  Py_complex cmpx = PyComplex_AsCComplex(item);
                  ((long double complex*)buffer)[i] = (long double)cmpx.real + (long double)cmpx.imag * I;
                }break;
      default: {
                  free(buffer);
                  PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
                  return NULL;
               }
    }
    if(PyErr_Occurred()){
      free(buffer);
      return NULL;
    }
  }
  PyObject* capsule = PyCapsule_New(buffer, "host_memory", free_memory);
  if(!capsule){
    free(capsule);
    return NULL;
  }
  return capsule;
}

static PyMethodDef methods[] = {
  {"array", array, METH_VARARGS, "allocate the memory for a list and return a capsule"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "host",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_host(void){
  return PyModule_Create(&module);
}

