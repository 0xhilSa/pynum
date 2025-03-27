#include <python3.10/Python.h>
#include <python3.10/abstract.h>
#include <python3.10/complexobject.h>
#include <python3.10/floatobject.h>
#include <python3.10/listobject.h>
#include <python3.10/longobject.h>
#include <python3.10/methodobject.h>
#include <python3.10/modsupport.h>
#include <python3.10/object.h>
#include <python3.10/pycapsule.h>
#include <python3.10/pyerrors.h>
#include <python3.10/pyport.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <complex.h>
#include <string.h>


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

static PyObject* toList(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &length, &fmt)) return NULL;
  void* buffer = PyCapsule_GetPointer(capsule, "host_memory");
  if(!buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }
  PyObject* py_list = PyList_New(length);
  if(!py_list){ return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject* item = NULL;
    switch (*fmt) {
      case '?': item = PyBool_FromLong(((bool*)buffer)[i]); break;
      case 'b': item = PyLong_FromLong(((char*)buffer)[i]); break;
      case 'B': item = PyLong_FromUnsignedLong(((unsigned char*)buffer)[i]); break;
      case 'h': item = PyLong_FromLong(((short*)buffer)[i]); break;
      case 'H': item = PyLong_FromUnsignedLong(((unsigned short*)buffer)[i]); break;
      case 'i': item = PyLong_FromLong(((int*)buffer)[i]); break;
      case 'I': item = PyLong_FromUnsignedLong(((unsigned int*)buffer)[i]); break;
      case 'l': item = PyLong_FromLong(((long*)buffer)[i]); break;
      case 'L': item = PyLong_FromUnsignedLong(((unsigned long*)buffer)[i]); break;
      case 'q': item = PyLong_FromLongLong(((long long*)buffer)[i]); break;
      case 'Q': item = PyLong_FromUnsignedLongLong(((unsigned long long*)buffer)[i]); break;
      case 'f': item = PyFloat_FromDouble(((float*)buffer)[i]); break;
      case 'd': item = PyFloat_FromDouble(((double*)buffer)[i]); break;
      case 'g': item = PyFloat_FromDouble(((long double*)buffer)[i]); break;
      case 'F': {
        float complex val = ((float complex*)buffer)[i];
        item = PyComplex_FromDoubles(crealf(val), cimagf(val));
      } break;
      case 'D': {
        double complex val = ((double complex*)buffer)[i];
        item = PyComplex_FromDoubles(creal(val), cimag(val));
      } break;
      case 'G': {
        long double complex val = ((long double complex*)buffer)[i];
        item = PyComplex_FromDoubles(creall(val), cimagl(val));
      } break;
      default: {
        PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
        Py_DECREF(py_list);
        return NULL;
      }
    }
    if(!item){
      Py_DECREF(py_list);
      return NULL;
    }
    PyList_SET_ITEM(py_list, i, item);
  }
  return py_list;
}

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
                  PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
                  return NULL;
               }
    }
    if(PyErr_Occurred()){
      printf("An Error Occurred at `array` func in host.c\n");
      fflush(stdout);
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

static PyObject* getitem_index(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length, index;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Onns", &capsule, &length, &index, &fmt)) return NULL;

  void* buffer = PyCapsule_GetPointer(capsule, "host_memory");
  if(!buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }

  if(!PyList_Check(buffer)){
    PyErr_SetString(PyExc_TypeError, "Expected a python list");
    return NULL;
  }

  if(index < 0) index += length;
  if(index < 0 || index >= length){
    PyErr_SetString(PyExc_IndexError, "Index out of bounds!");
    return NULL;
  }

  void* value = NULL;
  size_t type_size = 0;

  switch(*fmt){
    case '?': type_size = sizeof(bool); break;
    case 'b': type_size = sizeof(char); break;
    case 'B': type_size = sizeof(unsigned char); break;
    case 'h': type_size = sizeof(short); break;
    case 'H': type_size = sizeof(unsigned short); break;
    case 'i': type_size = sizeof(int); break;
    case 'I': type_size = sizeof(unsigned int); break;
    case 'l': type_size = sizeof(long); break;
    case 'L': type_size = sizeof(unsigned long); break;
    case 'q': type_size = sizeof(long long); break;
    case 'Q': type_size = sizeof(unsigned long long); break;
    case 'f': type_size = sizeof(float); break;
    case 'd': type_size = sizeof(double); break;
    case 'g': type_size = sizeof(long double); break;
    default:
      PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
      return NULL;
  }

  value = malloc(type_size);
  if (!value) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return NULL;
  }
  memcpy(value, (char*)buffer + index * type_size, type_size);

  PyObject* value_capsule = PyCapsule_New(value, "host_memory", free_memory);
  if(!value_capsule){
    free(value);
    PyErr_SetString(PyExc_RuntimeError, "Failed to create PyCapsule");
    return NULL;
  }
  return value_capsule;
}

static PyObject* getitem_slice(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length, start, stop, step;
  const char* fmt;

  if(!PyArg_ParseTuple(args, "Onnnns", &capsule, &length, &start, &stop, &step, &fmt)) return NULL;

  void* buffer = PyCapsule_GetPointer(capsule, "host_memory");
  if(!buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }

  Py_ssize_t sliced_len = (stop - start + step - (step > 0 ? 1 : -1)) / step;
  if(sliced_len <= 0) sliced_len = 0;

  size_t item_size = get_type_size(fmt);
  if(item_size == 0){
    PyErr_SetString(PyExc_TypeError, "Unsupported data type");
    return NULL;
  }

  void* sliced_buffer = malloc(sliced_len * item_size);
  if(!sliced_buffer){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate slice buffer");
    return NULL;
  }

  for(Py_ssize_t i = 0, j = start * item_size; i < sliced_len; i++, j += step * item_size){
    memcpy(sliced_buffer + i * item_size, buffer + j, item_size);
  }

  PyObject* result_capsule = PyCapsule_New(sliced_buffer, "host_memory", free_memory);
  if(!result_capsule){
    free(sliced_buffer);
    PyErr_SetString(PyExc_RuntimeError, "Failed to create PyCapsule");
    return NULL;
  }
  return result_capsule;
}

static PyObject* setitem_index(PyObject* self, PyObject* args){
  PyObject *capsule, *value;
  Py_ssize_t length, index;
  const char* fmt;

  if(!PyArg_ParseTuple(args, "OOnns", &capsule, &value, &length, &index, &fmt)) return NULL;

  void* buffer = PyCapsule_GetPointer(capsule, "host_memory");
  if(!buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }

  if(index < 0) index += length;
  if(index < 0 || index >= length){
    PyErr_SetString(PyExc_IndexError, "Index out of bounds!");
    return NULL;
  }

  switch (*fmt){
    case '?': ((bool*)buffer)[index] = PyObject_IsTrue(value); break;
    case 'b': ((char*)buffer)[index] = (char)PyLong_AsLong(value); break;
    case 'B': ((unsigned char*)buffer)[index] = (unsigned char)PyLong_AsUnsignedLong(value); break;
    case 'h': ((short*)buffer)[index] = (short)PyLong_AsLong(value); break;
    case 'H': ((unsigned short*)buffer)[index] = (unsigned short)PyLong_AsUnsignedLong(value); break;
    case 'i': ((int*)buffer)[index] = (int)PyLong_AsLong(value); break;
    case 'I': ((unsigned int*)buffer)[index] = (unsigned int)PyLong_AsUnsignedLong(value); break;
    case 'l': ((long*)buffer)[index] = (long)PyLong_AsLong(value); break;
    case 'L': ((unsigned long*)buffer)[index] = (unsigned long)PyLong_AsUnsignedLong(value); break;
    case 'q': ((long long*)buffer)[index] = (long long)PyLong_AsLongLong(value); break;
    case 'Q': ((unsigned long long*)buffer)[index] = (unsigned long long)PyLong_AsUnsignedLongLong(value); break;
    case 'f': ((float*)buffer)[index] = (float)PyFloat_AsDouble(value); break;
    case 'd': ((double*)buffer)[index] = PyFloat_AsDouble(value); break;
    case 'g': ((long double*)buffer)[index] = (long double)PyFloat_AsDouble(value); break;
    case 'F': {
                Py_complex cmpx = PyComplex_AsCComplex(value);
                ((float complex*)buffer)[index] = (float)cmpx.real + (float)cmpx.imag * I;
              } break;
    case 'D': {
                Py_complex cmpx = PyComplex_AsCComplex(value);
                ((double complex*)buffer)[index] = (double)cmpx.real + (double)cmpx.imag * I;
              } break;
    case 'G': {
                Py_complex cmpx = PyComplex_AsCComplex(value);
                ((long double complex*)buffer)[index] = (long double)cmpx.real + (long double)cmpx.imag * I;
              } break;
    default:
      PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
      return NULL;
  }

  if(PyErr_Occurred()){
    printf("An Error Occurred at `setitem_index` func in host.c\n");
    fflush(stdout);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* setitem_slice(PyObject* self, PyObject* args){
  PyObject *capsule, *values;
  Py_ssize_t length, start, stop, step;
  const char* fmt;

  if(!PyArg_ParseTuple(args, "OOnnnns", &capsule, &values, &length, &start, &stop, &step, &fmt)) return NULL;

  void* buffer = PyCapsule_GetPointer(capsule, "host_memory");
  if(!buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid memory capsule");
    return NULL;
  }

  Py_ssize_t sliced_len = (stop - start + step - (step > 0 ? 1 : -1)) / step;
  if(sliced_len <= 0) sliced_len = 0;

  if(!PyList_Check(values)){
    PyErr_SetString(PyExc_TypeError, "Slice values must be a list");
    return NULL;
  }

  if(PyList_Size(values) != sliced_len){
    PyErr_SetString(PyExc_ValueError, "Slice values length does not match slice");
    return NULL;
  }

  size_t item_size = get_type_size(fmt);
  if(item_size == 0){
    PyErr_SetString(PyExc_TypeError, "Unsupported data type");
    return NULL;
  }
  
  for(Py_ssize_t i = 0; i < sliced_len; i++){
    PyObject* value = PyList_GetItem(values, i);
    Py_ssize_t index = start + i * step;

    switch (*fmt){
      case '?': ((bool*)buffer)[index] = PyObject_IsTrue(value); break;
      case 'b': ((char*)buffer)[index] = (char)PyLong_AsLong(value); break;
      case 'B': ((unsigned char*)buffer)[index] = (unsigned char)PyLong_AsUnsignedLong(value); break;
      case 'h': ((short*)buffer)[index] = (short)PyLong_AsLong(value); break;
      case 'H': ((unsigned short*)buffer)[index] = (unsigned short)PyLong_AsUnsignedLong(value); break;
      case 'i': ((int*)buffer)[index] = (int)PyLong_AsLong(value); break;
      case 'I': ((unsigned int*)buffer)[index] = (unsigned int)PyLong_AsUnsignedLong(value); break;
      case 'l': ((long*)buffer)[index] = (long)PyLong_AsLong(value); break;
      case 'L': ((unsigned long*)buffer)[index] = (unsigned long)PyLong_AsUnsignedLong(value); break;
      case 'q': ((long long*)buffer)[index] = (long long)PyLong_AsLongLong(value); break;
      case 'Q': ((unsigned long long*)buffer)[index] = (unsigned long long)PyLong_AsUnsignedLongLong(value); break;
      case 'f': ((float*)buffer)[index] = (float)PyFloat_AsDouble(value); break;
      case 'd': ((double*)buffer)[index] = PyFloat_AsDouble(value); break;
      case 'g': ((long double*)buffer)[index] = (long double)PyFloat_AsDouble(value); break;
      case 'F': {
                  Py_complex cmpx = PyComplex_AsCComplex(value);
                  ((float complex*)buffer)[index] = (float)cmpx.real + (float)cmpx.imag * I;
                } break;
      case 'D': {
                  Py_complex cmpx = PyComplex_AsCComplex(value);
                  ((double complex*)buffer)[index] = (double)cmpx.real + (double)cmpx.imag * I;
                } break;
      case 'G': {
                  Py_complex cmpx = PyComplex_AsCComplex(value);
                  ((long double complex*)buffer)[index] = (long double)cmpx.real + (long double)cmpx.imag * I;
                } break;
      default:
        PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
        return NULL;
    }

    if(PyErr_Occurred()){
      printf("An Error Occurred at `setitem_slice` func in host.c\n");
      fflush(stdout);
      return NULL;
    }
  }
  Py_RETURN_NONE;
}



void add_vector_kernel(void* result, void* buff1, void* buff2, size_t length, const char* fmt){
  if(strcmp(fmt, "?") == 0){
    bool* x = (bool*)buff1;
    bool* y = (bool*)buff2;
    bool* z = (bool*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] | y[i]; }
  }else if(strcmp(fmt, "b") == 0){
    char* x = (char*)buff1;
    char* y = (char*)buff2;
    char* z = (char*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "B") == 0){
    unsigned char* x = (unsigned char*)buff1;
    unsigned char* y = (unsigned char*)buff2;
    unsigned char* z = (unsigned char*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "h") == 0){
    short* x = (short*)buff1;
    short* y = (short*)buff2;
    short* z = (short*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "H") == 0){
    unsigned short* x = (unsigned short*)buff1;
    unsigned short* y = (unsigned short*)buff2;
    unsigned short* z = (unsigned short*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "i") == 0){
    int* x = (int*)buff1;
    int* y = (int*)buff2;
    int* z = (int*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "I") == 0){
    unsigned int* x = (unsigned int*)buff1;
    unsigned int* y = (unsigned int*)buff2;
    unsigned int* z = (unsigned int*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "l") == 0){
    long* x = (long*)buff1;
    long* y = (long*)buff2;
    long* z = (long*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "L") == 0){
    unsigned long* x = (unsigned long*)buff1;
    unsigned long* y = (unsigned long*)buff2;
    unsigned long* z = (unsigned long*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "q") == 0){
    long long* x = (long long*)buff1;
    long long* y = (long long*)buff2;
    long long* z = (long long*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "Q") == 0){
    unsigned long long* x = (unsigned long long*)buff1;
    unsigned long long* y = (unsigned long long*)buff2;
    unsigned long long* z = (unsigned long long*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "f") == 0){
    float* x = (float*)buff1;
    float* y = (float*)buff2;
    float* z = (float*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "d") == 0){
    double* x = (double*)buff1;
    double* y = (double*)buff2;
    double* z = (double*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "g") == 0){
    long double* x = (long double*)buff1;
    long double* y = (long double*)buff2;
    long double* z = (long double*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "F") == 0){
    float complex* x = (float complex*)buff1;
    float complex* y = (float complex*)buff2;
    float complex* z = (float complex*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "D") == 0){
    double complex* x = (double complex*)buff1;
    double complex* y = (double complex*)buff2;
    double complex* z = (double complex*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }else if(strcmp(fmt, "G") == 0){
    long double complex* x = (long double complex*)buff1;
    long double complex* y = (long double complex*)buff2;
    long double complex* z = (long double complex*)result;
    for(Py_ssize_t i = 0; i < length; i++){ z[i] = x[i] + y[i]; }
  }
}

static PyObject* add_vector(PyObject* self, PyObject* args){
  PyObject *vec1, *vec2;
  Py_ssize_t length;
  const char *fmt1, *fmt2;

  if(!PyArg_ParseTuple(args, "OOnss", &vec1, &vec2, &length, &fmt1, &fmt2)) return NULL;

  if(strcmp(fmt1, fmt2) != 0){
    PyErr_SetString(PyExc_TypeError, "DType of vectors must be the same");
    return NULL;
  }
  void* buff1 = PyCapsule_GetPointer(vec1, "host_memory");
  void* buff2 = PyCapsule_GetPointer(vec2, "host_memory");
  if(!buff1 || !buff2){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule: NULL pointer");
    return NULL;
  }

  size_t size =  get_type_size(fmt1);
  void* result = malloc(length * sizeof(size));
  if(!result){
    PyErr_NoMemory();
    return NULL;
  }

  add_vector_kernel(result, buff1, buff2, length, fmt1);
  return PyCapsule_New(result, "host_memory", free_memory);
}


static PyMethodDef methods[] = {
  {"array", array, METH_VARARGS, "allocate the memory for a list and return a capsule"},
  {"toList", toList, METH_VARARGS, "from pycapsule to python list"},
  {"getitem_index", getitem_index, METH_VARARGS, "get item through index"},
  {"getitem_slice", getitem_slice, METH_VARARGS, "get item through slice"},
  {"setitem_index", setitem_index, METH_VARARGS, "set item through index"},
  {"setitem_slice", setitem_slice, METH_VARARGS, "set item through slice"},
  {"add_vector", add_vector, METH_VARARGS, "add 2 vectors"},
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
