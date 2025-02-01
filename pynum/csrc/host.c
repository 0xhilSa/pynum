#include <Python.h>
#include <complex.h>
#include <stdbool.h>


static void free_capsule(PyObject* capsule){
  void* ptr = PyCapsule_GetPointer(capsule, "array_pointer");
  if(ptr){ free(ptr); }
}

static PyObject* py_list(PyObject* self, PyObject* args){
  PyObject* py_list;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &py_list, &fmt)) return NULL;
  if(!PyList_Check(py_list)){
    PyErr_SetString(PyExc_TypeError, "Expected a Python List");
    return NULL;
  }
  size_t length = PyList_Size(py_list);
  void* array = NULL;
  if(strcmp(fmt, "b") == 0){
    array = malloc(length * sizeof(char));
    for(size_t i = 0; i < length; i++){ ((char*)array)[i] = (char)PyLong_AsLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "B") == 0){
    array = malloc(length * sizeof(unsigned char));
    for(size_t i = 0; i < length; i++){ ((unsigned char*)array)[i] = (unsigned char)PyLong_AsUnsignedLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "h") == 0){
    array = malloc(length * sizeof(short));
    for(size_t i = 0; i < length; i++){ ((short*)array)[i] = (short)PyLong_AsLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "H") == 0){
    array = malloc(length * sizeof(unsigned short));
    for(size_t i = 0; i < length; i++){ ((unsigned short*)array)[i] = (unsigned short)PyLong_AsUnsignedLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "i") == 0){
    array = malloc(length * sizeof(int));
    for(size_t i = 0; i < length; i++){ ((int*)array)[i] = (int)PyLong_AsLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "I") == 0){
    array = malloc(length * sizeof(unsigned int));
    for(size_t i = 0; i < length; i++){ ((unsigned int*)array)[i] = (unsigned int)PyLong_AsUnsignedLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "l") == 0){
    array = malloc(length * sizeof(long));
    for(size_t i = 0; i < length; i++){ ((long*)array)[i] = PyLong_AsLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "L") == 0){
    array = malloc(length * sizeof(unsigned long));
    for(size_t i = 0; i < length; i++){ ((unsigned long*)array)[i] = (unsigned long)PyLong_AsUnsignedLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "q") == 0){
    array = malloc(length * sizeof(long long));
    for(size_t i = 0; i < length; i++){ ((long long*)array)[i] = (long long)PyLong_AsLongLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "Q") == 0){
    array = malloc(length * sizeof(unsigned long long));
    for(size_t i = 0; i < length; i++){ ((unsigned long long*)array)[i] = (unsigned long long)PyLong_AsUnsignedLongLong(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "f") == 0){
    array = malloc(length * sizeof(float));
    for(size_t i = 0; i < length; i++){ ((float*)array)[i] = (float)PyFloat_AsDouble(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "d") == 0){
    array = malloc(length * sizeof(double));
    for(size_t i = 0; i < length; i++){ ((double*)array)[i] = PyFloat_AsDouble(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "g") == 0){
    array = malloc(length * sizeof(long double));
    for(size_t i = 0; i < length; i++){ ((long double*)array)[i] = (long double)PyFloat_AsDouble(PyList_GetItem(py_list, i)); }
  }else if(strcmp(fmt, "F") == 0){
    array = malloc(length * sizeof(float complex));
    for(size_t i = 0; i < length; i++){
      PyObject* item = PyList_GetItem(py_list, i);
      ((float complex*)array)[i] = (float)PyComplex_RealAsDouble(item) + (float)PyComplex_ImagAsDouble(item) * I;
    }
  }else if(strcmp(fmt, "D") == 0){
    array = malloc(length * sizeof(double complex));
    for(size_t i = 0; i < length; i++){
      PyObject* item = PyList_GetItem(py_list, i);
      ((double complex*)array)[i] = (double)PyComplex_RealAsDouble(item) + (double)PyComplex_ImagAsDouble(item) * I;
    }
  }else if(strcmp(fmt, "G") == 0){
    array = malloc(length * sizeof(long double complex));
    for(size_t i = 0; i < length; i++){
      PyObject* item = PyList_GetItem(py_list, i);
      ((long double complex*)array)[i] = (long double)PyComplex_RealAsDouble(item) + (long double)PyComplex_ImagAsDouble(item) * I;
    }
  }else if(strcmp(fmt, "?") == 0){
    array = malloc(length * sizeof(bool));
    for(size_t i = 0; i < length; i++){ ((bool*)array)[i] = PyObject_IsTrue(PyList_GetItem(py_list, i)); }
  }else{
    free(array);
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s' provided", fmt);
    return NULL;
  }
  if(!array){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory"); }
  return PyCapsule_New(array, "array_pointer", free_capsule);
}

static PyObject* py_list_from_capsule(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &length, &fmt)){ return NULL; }
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule");
    return NULL;
  }
  void* array = PyCapsule_GetPointer(capsule, "array_pointer");
  if(!array){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  PyObject* py_list = PyList_New(length);
  if(!py_list){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate Python list");
    return NULL;
  }
  for(Py_ssize_t i = 0; i < length; i++){
    if(strcmp(fmt, "b") == 0){ PyList_SetItem(py_list, i, PyLong_FromLong(((char*)array)[i])); }
    else if(strcmp(fmt, "B") == 0){ PyList_SetItem(py_list, i, PyLong_FromUnsignedLong(((unsigned char*)array)[i])); }
    else if(strcmp(fmt, "h") == 0){ PyList_SetItem(py_list, i, PyLong_FromLong(((short*)array)[i])); }
    else if(strcmp(fmt, "H") == 0){ PyList_SetItem(py_list, i, PyLong_FromUnsignedLong(((unsigned short*)array)[i])); }
    else if(strcmp(fmt, "i") == 0){ PyList_SetItem(py_list, i, PyLong_FromLong(((int*)array)[i])); }
    else if(strcmp(fmt, "I") == 0){ PyList_SetItem(py_list, i, PyLong_FromUnsignedLong(((unsigned int*)array)[i])); }
    else if(strcmp(fmt, "l") == 0){ PyList_SetItem(py_list, i, PyLong_FromLong(((long*)array)[i])); }
    else if(strcmp(fmt, "L") == 0){ PyList_SetItem(py_list, i, PyLong_FromUnsignedLong(((unsigned long*)array)[i])); }
    else if(strcmp(fmt, "q") == 0){ PyList_SetItem(py_list, i, PyLong_FromLongLong(((long long*)array)[i])); }
    else if(strcmp(fmt, "Q") == 0){ PyList_SetItem(py_list, i, PyLong_FromUnsignedLongLong(((unsigned long long*)array)[i])); }
    else if(strcmp(fmt, "f") == 0){ PyList_SetItem(py_list, i, PyFloat_FromDouble(((float*)array)[i])); }
    else if(strcmp(fmt, "d") == 0){ PyList_SetItem(py_list, i, PyFloat_FromDouble(((double*)array)[i])); }
    else if(strcmp(fmt, "F") == 0){ PyList_SetItem(py_list, i, PyComplex_FromDoubles(crealf(((float complex*)array)[i]), cimagf(((float complex*)array)[i]))); }
    else if(strcmp(fmt, "D") == 0){ PyList_SetItem(py_list, i, PyComplex_FromDoubles(creal(((double complex*)array)[i]), cimag(((double complex*)array)[i]))); }
    else if(strcmp(fmt, "G") == 0){ PyList_SetItem(py_list, i, PyComplex_FromDoubles(creall(((long double complex*)array)[i]), cimagl(((long double complex*)array)[i]))); }
    else if(strcmp(fmt, "?") == 0){ PyList_SetItem(py_list, i, PyBool_FromLong(((bool*)array)[i])); }
    else{
      PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
      Py_DECREF(py_list);
      return NULL;
    }
  }
  return py_list;
}

static PyObject* py_get_value(PyObject* self, PyObject* args){
  PyObject* capsule;
  Py_ssize_t index;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &capsule, &index, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(capsule)){
    PyErr_SetString(PyExc_TypeError, "Invalid PyCapsule");
    return NULL;
  }
  void* array = PyCapsule_GetPointer(capsule, "array_pointer");
  if(!array){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule");
    return NULL;
  }
  if(index < 0){
    PyErr_SetString(PyExc_IndexError, "Index cannot be negative");
    return NULL;
  }
  if(strcmp(fmt, "b") == 0) return PyLong_FromLong(((char*)array)[index]);
  else if(strcmp(fmt, "B") == 0) return PyLong_FromUnsignedLong(((unsigned char*)array)[index]);
  else if(strcmp(fmt, "h") == 0) return PyLong_FromLong(((short*)array)[index]);
  else if(strcmp(fmt, "H") == 0) return PyLong_FromLong(((unsigned short*)array)[index]);
  else if(strcmp(fmt, "i") == 0) return PyLong_FromLong(((int*)array)[index]);
  else if(strcmp(fmt, "I") == 0) return PyLong_FromUnsignedLong(((unsigned int*)array)[index]);
  else if(strcmp(fmt, "l") == 0) return PyLong_FromLong(((long*)array)[index]);
  else if(strcmp(fmt, "L") == 0) return PyLong_FromUnsignedLong(((unsigned long*)array)[index]);
  else if(strcmp(fmt, "q") == 0) return PyLong_FromLongLong(((long long*)array)[index]);
  else if(strcmp(fmt, "Q") == 0) return PyLong_FromUnsignedLongLong(((unsigned long long*)array)[index]);
  else if(strcmp(fmt, "f") == 0) return PyFloat_FromDouble(((float*)array)[index]);
  else if(strcmp(fmt, "d") == 0) return PyFloat_FromDouble(((double*)array)[index]);
  else if(strcmp(fmt, "g") == 0) return PyFloat_FromDouble(((long double*)array)[index]);
  else if(strcmp(fmt, "F") == 0) return PyComplex_FromDoubles(crealf(((float complex*)array)[index]), cimagf(((float complex*)array)[index]));
  else if(strcmp(fmt, "D") == 0) return PyComplex_FromDoubles(creal(((double complex*)array)[index]), cimag(((double complex*)array)[index]));
  else if(strcmp(fmt, "G") == 0) return PyComplex_FromDoubles(creall(((long double complex*)array)[index]), cimagl(((long double complex*)array)[index]));
  else if(strcmp(fmt, "?") == 0) return PyBool_FromLong(((bool*)array)[index]);
  else{
    PyErr_Format(PyExc_TypeError, "Invalid DType: '%s'", fmt);
    return NULL;
  }
}


static PyMethodDef Methods[] = {
  {"array", py_list, METH_VARARGS, "create an array with only one king of dtype"},
  {"toList", py_list_from_capsule, METH_VARARGS, "get python list from the the capsule"},
  {"get", py_get_value, METH_VARARGS, "get value by index"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef host_module = {
  PyModuleDef_HEAD_INIT,
  "host",
  "Extended func using C APIs",
  -1,
  Methods
};

PyMODINIT_FUNC PyInit_host(void){ return PyModule_Create(&host_module); }
