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
from .src.cuda_stream import py_alloc_bool

# copy from host to device
from .src.cuda_stream import py_memcpy_htod_short
from .src.cuda_stream import py_memcpy_htod_int
from .src.cuda_stream import py_memcpy_htod_long
from .src.cuda_stream import py_memcpy_htod_float
from .src.cuda_stream import py_memcpy_htod_double
from .src.cuda_stream import py_memcpy_htod_complex
from .src.cuda_stream import py_memcpy_htod_bool

# copy from host to host
from .src.cuda_stream import py_memcpy_htoh_short
from .src.cuda_stream import py_memcpy_htoh_int
from .src.cuda_stream import py_memcpy_htoh_long
from .src.cuda_stream import py_memcpy_htoh_float
from .src.cuda_stream import py_memcpy_htoh_double
from .src.cuda_stream import py_memcpy_htoh_complex
from .src.cuda_stream import py_memcpy_htoh_bool

# copy from device to host
from .src.cuda_stream import py_memcpy_dtoh_short
from .src.cuda_stream import py_memcpy_dtoh_int
from .src.cuda_stream import py_memcpy_dtoh_long
from .src.cuda_stream import py_memcpy_dtoh_float
from .src.cuda_stream import py_memcpy_dtoh_double
from .src.cuda_stream import py_memcpy_dtoh_complex
from .src.cuda_stream import py_memcpy_dtoh_bool

# copy from device to device
from .src.cuda_stream import py_memcpy_dtod_short
from .src.cuda_stream import py_memcpy_dtod_int
from .src.cuda_stream import py_memcpy_dtod_long
from .src.cuda_stream import py_memcpy_dtod_float
from .src.cuda_stream import py_memcpy_dtod_double
from .src.cuda_stream import py_memcpy_dtod_complex
from .src.cuda_stream import py_memcpy_dtod_bool

# getters(index)
from .src.cuda_stream import py_get_value_short
from .src.cuda_stream import py_get_value_int
from .src.cuda_stream import py_get_value_long
from .src.cuda_stream import py_get_value_float
from .src.cuda_stream import py_get_value_double
from .src.cuda_stream import py_get_value_complex
from .src.cuda_stream import py_get_value_bool

# getter(slice)
from .src.cuda_stream import py_get_slice_short
from .src.cuda_stream import py_get_slice_int
from .src.cuda_stream import py_get_slice_long
from .src.cuda_stream import py_get_slice_float
from .src.cuda_stream import py_get_slice_double
from .src.cuda_stream import py_get_slice_complex
from .src.cuda_stream import py_get_slice_bool

# setters(index)
from .src.cuda_stream import py_set_value_short
from .src.cuda_stream import py_set_value_int
from .src.cuda_stream import py_set_value_long
from .src.cuda_stream import py_set_value_float
from .src.cuda_stream import py_set_value_double
from .src.cuda_stream import py_set_value_complex
from .src.cuda_stream import py_set_value_bool

# freebie
from .src.cuda_stream import py_free
from .src.cuda_stream import py_query_free_memory

# count number of CUDA-device
from .src.cuda_stream import py_count_device


TYPE = [int, float, complex]

class POINTER:
  """
  POINTER(ptr) must have its own class to encapsulate the pointer from the other data types
  e.g. int, float, and complex.
  """
  def __init__(self, ptr: int): self.__ptr = ptr
  def __repr__(self): return f"PTR({self.__ptr})"
  def __del__(self): del self
  @property
  def value(self): return self.__ptr


def alloc_short(x:List): return POINTER(py_alloc_short(x))
def alloc_int(x:List): return POINTER(py_alloc_int(x))
def alloc_long(x:List): return POINTER(py_alloc_long(x))
def alloc_float(x:List): return POINTER(py_alloc_float(x))
def alloc_double(x:List): return POINTER(py_alloc_double(x))
def alloc_complex(x:List): return POINTER(py_alloc_complex(x))
# def alloc_bool(x:List): return POINTER(py_alloc_bool(x))

def free(ptr:POINTER): py_free(ptr.value); return None
def query_free_memory(): return py_query_free_memory()

def memcpy_htod_short(ptr:POINTER, x:List): py_memcpy_htod_short(ptr.value, x); return None
def memcpy_htod_int(ptr:POINTER, x:List): py_memcpy_htod_int(ptr.value, x); return None
def memcpy_htod_long(ptr:POINTER, x:List): py_memcpy_htod_long(ptr.value, x); return None
def memcpy_htod_float(ptr:POINTER, x:List): py_memcpy_htod_float(ptr.value, x); return None
def memcpy_htod_double(ptr:POINTER, x:List): py_memcpy_htod_double(ptr.value, x); return None
def memcpy_htod_complex(ptr:POINTER, x:List): py_memcpy_htod_complex(ptr.value, x); return None
# def memcpy_htod_bool(ptr:POINTER, x:List): py_memcpy_htod_bool(ptr.value, x); return None

def memcpy_htoh_short(src:List, dst:List): py_memcpy_htoh_short(src, dst); return None
def memcpy_htoh_int(src:List, dst:List): py_memcpy_htoh_int(src, dst); return None
def memcpy_htoh_long(src:List, dst:List): py_memcpy_htoh_long(src, dst); return None
def memcpy_htoh_float(src:List, dst:List): py_memcpy_htoh_float(src, dst); return None
def memcpy_htoh_double(src:List, dst:List): py_memcpy_htoh_double(src, dst); return None
def memcpy_htoh_complex(src:List, dst:List): py_memcpy_htoh_complex(src, dst); return None
# def memcpy_htod_bool(ptr:POINTER, x:List): py_memcpy_htod_bool(ptr.value, x); return None

def memcpy_dtoh_short(ptr:POINTER, length:int): return py_memcpy_dtoh_short(ptr.value, length)
def memcpy_dtoh_int(ptr:POINTER, length:int): return py_memcpy_dtoh_int(ptr.value, length)
def memcpy_dtoh_long(ptr:POINTER, length:int): return py_memcpy_dtoh_long(ptr.value, length)
def memcpy_dtoh_float(ptr:POINTER, length:int): return py_memcpy_dtoh_float(ptr.value, length)
def memcpy_dtoh_double(ptr:POINTER, length:int): return py_memcpy_dtoh_double(ptr.value, length)
def memcpy_dtoh_complex(ptr:POINTER, length:int): return py_memcpy_dtoh_complex(ptr.value, length)
# def memcpy_dtoh_bool(ptr:POINTER, length:int): return py_memcpy_dtoh_bool(ptr.value, length)

def memcpy_dtod_short(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_short(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_int(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_int(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_long(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_long(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_float(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_float(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_double(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_double(src_ptr.value, dst_ptr.value, length); return None
def memcpy_dtod_complex(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_complex(src_ptr.value, dst_ptr.value, length); return None
# def memcpy_dtod_bool(src_ptr:POINTER, dst_ptr:POINTER, length:int): py_memcpy_dtod_bool(src_ptr.value, dst_ptr.value, length); return None

def get_value_short(ptr:POINTER, index:int): return py_get_value_short(ptr.value, index)
def get_value_int(ptr:POINTER, index:int): return py_get_value_int(ptr.value, index)
def get_value_long(ptr:POINTER, index:int): return py_get_value_long(ptr.value, index)
def get_value_float(ptr:POINTER, index:int): return py_get_value_float(ptr.value, index)
def get_value_double(ptr:POINTER, index:int): return py_get_value_double(ptr.value, index)
def get_value_complex(ptr:POINTER, index:int): return py_get_value_complex(ptr.value, index)
# def get_value_bool(ptr:POINTER, index:int): return py_get_value_bool(ptr.value, index)

def get_slice_short(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_short(ptr.value, start, stop, step)
def get_slice_int(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_int(ptr.value, start, stop, step)
def get_slice_long(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_long(ptr.value, start, stop, step)
def get_slice_float(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_float(ptr.value, start, stop, step)
def get_slice_double(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_double(ptr.value, start, stop, step)
def get_slice_complex(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_complex(ptr.value, start, stop, step)
# def get_slice_bool(ptr:POINTER, start:int, stop:int, step:int): return py_get_slice_bool(ptr.value, start, stop, step)

def set_value_short(ptr:POINTER, index:int): py_set_value_short(ptr.value, index); return None
def set_value_int(ptr:POINTER, index:int): py_set_value_int(ptr.value, index); return None
def set_value_long(ptr:POINTER, index:int): py_set_value_long(ptr.value, index); return None
def set_value_float(ptr:POINTER, index:int): py_set_value_float(ptr.value, index); return None
def set_value_double(ptr:POINTER, index:int): py_set_value_double(ptr.value, index); return None
def set_value_complex(ptr:POINTER, index:int): py_set_value_complex(ptr.value, index); return None
# def set_value_bool(ptr:POINTER, index:int): py_set_value_bool(ptr.value, index); return None

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