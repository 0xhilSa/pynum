"""
these functions are written in C and
defined at './pynum/src/cuda_stream.c'
"""

from typing import Union, Type, List
import subprocess

# allocators for different data types
from .src.cuda_stream import py_alloc_short
from .src.cuda_stream import py_alloc_int
from .src.cuda_stream import py_alloc_long
from .src.cuda_stream import py_alloc_float
from .src.cuda_stream import py_alloc_double
from .src.cuda_stream import py_alloc_complex

# copy from host to device
from .src.cuda_stream import py_memcpy_htod_short
from .src.cuda_stream import py_memcpy_htod_int
from .src.cuda_stream import py_memcpy_htod_long
from .src.cuda_stream import py_memcpy_htod_float
from .src.cuda_stream import py_memcpy_htod_double
from .src.cuda_stream import py_memcpy_htod_complex

# copy from host to host
from .src.cuda_stream import py_memcpy_htoh_short
from .src.cuda_stream import py_memcpy_htoh_int
from .src.cuda_stream import py_memcpy_htoh_long
from .src.cuda_stream import py_memcpy_htoh_float
from .src.cuda_stream import py_memcpy_htoh_double
from .src.cuda_stream import py_memcpy_htoh_complex

# copy from device to host
from .src.cuda_stream import py_memcpy_dtoh_short
from .src.cuda_stream import py_memcpy_dtoh_int
from .src.cuda_stream import py_memcpy_dtoh_long
from .src.cuda_stream import py_memcpy_dtoh_float
from .src.cuda_stream import py_memcpy_dtoh_double
from .src.cuda_stream import py_memcpy_dtoh_complex

# copy from device to device
from .src.cuda_stream import py_memcpy_dtod_short
from .src.cuda_stream import py_memcpy_dtod_int
from .src.cuda_stream import py_memcpy_dtod_long
from .src.cuda_stream import py_memcpy_dtod_float
from .src.cuda_stream import py_memcpy_dtod_double
from .src.cuda_stream import py_memcpy_dtod_complex

# getters(index)
from .src.cuda_stream import py_get_value_short
from .src.cuda_stream import py_get_value_int
from .src.cuda_stream import py_get_value_long
from .src.cuda_stream import py_get_value_float
from .src.cuda_stream import py_get_value_double
from .src.cuda_stream import py_get_value_complex

# getter(slice)
from .src.cuda_stream import py_get_slice_short
from .src.cuda_stream import py_get_slice_int
from .src.cuda_stream import py_get_slice_long
from .src.cuda_stream import py_get_slice_float
from .src.cuda_stream import py_get_slice_double
from .src.cuda_stream import py_get_slice_complex

# setters(index)
from .src.cuda_stream import py_set_value_short
from .src.cuda_stream import py_set_value_int
from .src.cuda_stream import py_set_value_long
from .src.cuda_stream import py_set_value_float
from .src.cuda_stream import py_set_value_double
from .src.cuda_stream import py_set_value_complex

# freebie
from .src.cuda_stream import py_free
from .src.cuda_stream import py_query_free_memory

# count number of CUDA-device
from .src.cuda_stream import py_count_device


TYPE = [int, float, complex]

class __POINTER:
  """
  POINTER(ptr) must have its own class to encapsulate the pointer from the other data types
  e.g. int, float, and complex.
  """
  def __init__(self, ptr: int): self.__ptr = ptr
  def __repr__(self): return f"PTR({self.__ptr})"
  @property
  def value(self): return self.__ptr


def alloc_short(x:List): return __POINTER(py_alloc_short(x))
def alloc_int(x:List): return __POINTER(py_alloc_int(x))
def alloc_long(x:List): return __POINTER(py_alloc_long(x))
def alloc_float(x:List): return __POINTER(py_alloc_float(x))
def alloc_double(x:List): return __POINTER(py_alloc_double(x))
def alloc_complex(x:List): return __POINTER(py_alloc_complex(x))

def free(ptr:__POINTER): py_free(ptr.value); return None
def query_free_memory(): return py_query_free_memory()

def memcpy_htod_short(ptr:__POINTER, x:List): py_memcpy_htod_short(ptr.value, x); return None
def memcpy_htod_int(ptr:__POINTER, x:List): py_memcpy_htod_int(ptr.value, x); return None
def memcpy_htod_long(ptr:__POINTER, x:List): py_memcpy_htod_long(ptr.value, x); return None
def memcpy_htod_float(ptr:__POINTER, x:List): py_memcpy_htod_float(ptr.value, x); return None
def memcpy_htod_double(ptr:__POINTER, x:List): py_memcpy_htod_double(ptr.value, x); return None
def memcpy_htod_complex(ptr:__POINTER, x:List): py_memcpy_htod_complex(ptr.value, x); return None

def memcpy_htoh_short(src:List, dst:List): py_memcpy_htoh_short(src, dst); return None
def memcpy_htoh_int(src:List, dst:List): py_memcpy_htoh_int(src, dst); return None
def memcpy_htoh_long(src:List, dst:List): py_memcpy_htoh_long(src, dst); return None
def memcpy_htoh_float(src:List, dst:List): py_memcpy_htoh_float(src, dst); return None
def memcpy_htoh_double(src:List, dst:List): py_memcpy_htoh_double(src, dst); return None
def memcpy_htoh_complex(src:List, dst:List): py_memcpy_htoh_complex(src, dst); return None

def memcpy_dtoh_short(ptr:__POINTER, length:int): return py_memcpy_dtoh_short(ptr.value, length)
def memcpy_dtoh_int(ptr:__POINTER, length:int): return py_memcpy_dtoh_int(ptr.value, length)
def memcpy_dtoh_long(ptr:__POINTER, length:int): return py_memcpy_dtoh_long(ptr.value, length)
def memcpy_dtoh_float(ptr:__POINTER, length:int): return py_memcpy_dtoh_float(ptr.value, length)
def memcpy_dtoh_double(ptr:__POINTER, length:int): return py_memcpy_dtoh_double(ptr.value, length)
def memcpy_dtoh_complex(ptr:__POINTER, length:int): return py_memcpy_dtoh_complex(ptr.value, length)

def memcpy_dtod_short(src_ptr:__POINTER, dst_ptr:__POINTER, length:int): py_memcpy_dtod_short(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_int(src_ptr:__POINTER, dst_ptr:__POINTER, length:int): py_memcpy_dtod_int(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_long(src_ptr:__POINTER, dst_ptr:__POINTER, length:int): py_memcpy_dtod_long(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_float(src_ptr:__POINTER, dst_ptr:__POINTER, length:int): py_memcpy_dtod_float(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_double(src_ptr:__POINTER, dst_ptr:__POINTER, length:int): py_memcpy_dtod_double(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_complex(src_ptr:__POINTER, dst_ptr:__POINTER, length:int): py_memcpy_dtod_complex(src_ptr.value, dst_ptr.value, length); return None

def get_value_short(ptr:__POINTER, index:int): return py_get_value_short(ptr.value, index)
def get_value_int(ptr:__POINTER, index:int): return py_get_value_int(ptr.value, index)
def get_value_long(ptr:__POINTER, index:int): return py_get_value_long(ptr.value, index)
def get_value_float(ptr:__POINTER, index:int): return py_get_value_float(ptr.value, index)
def get_value_double(ptr:__POINTER, index:int): return py_get_value_double(ptr.value, index)
def get_value_complex(ptr:__POINTER, index:int): return py_get_value_complex(ptr.value, index)

def get_slice_short(ptr:__POINTER, start:int, stop:int, step:int): return py_get_slice_short(ptr.value, start, stop, step)
def get_slice_int(ptr:__POINTER, start:int, stop:int, step:int): return py_get_slice_int(ptr.value, start, stop, step)
def get_slice_long(ptr:__POINTER, start:int, stop:int, step:int): return py_get_slice_long(ptr.value, start, stop, step)
def get_slice_float(ptr:__POINTER, start:int, stop:int, step:int): return py_get_slice_float(ptr.value, start, stop, step)
def get_slice_double(ptr:__POINTER, start:int, stop:int, step:int): return py_get_slice_double(ptr.value, start, stop, step)
def get_slice_complex(ptr:__POINTER, start:int, stop:int, step:int): return py_get_slice_complex(ptr.value, start, stop, step)

def set_value_short(ptr:__POINTER, index:int): py_set_value_short(ptr.value, index); return None
def set_value_int(ptr:__POINTER, index:int): py_set_value_int(ptr.value, index); return None
def set_value_long(ptr:__POINTER, index:int): py_set_value_long(ptr.value, index); return None
def set_value_float(ptr:__POINTER, index:int): py_set_value_float(ptr.value, index); return None
def set_value_double(ptr:__POINTER, index:int): py_set_value_double(ptr.value, index); return None
def set_value_complex(ptr:__POINTER, index:int): py_set_value_complex(ptr.value, index); return None

def count_device(): return py_count_device()

# check for CUDA device availability
def is_available():
  try:
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0
  except FileNotFoundError: pass
  try:
    result = subprocess.run(["nvcc","--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0
  except FileNotFoundError: pass
  return False
