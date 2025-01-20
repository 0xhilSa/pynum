"""
these functions are written in CUDA and
defined at './pynum/src/cuda_stream.cu'
"""

from typing import List
import subprocess

# allocators for different data types
from .src.cuda_stream import alloc_short
from .src.cuda_stream import alloc_int
from .src.cuda_stream import alloc_long
from .src.cuda_stream import alloc_float
from .src.cuda_stream import alloc_double
from .src.cuda_stream import alloc_complex
from .src.cuda_stream import alloc_bool

# copy from host to device
from .src.cuda_stream import memcpy_htod_short
from .src.cuda_stream import memcpy_htod_int
from .src.cuda_stream import memcpy_htod_long
from .src.cuda_stream import memcpy_htod_float
from .src.cuda_stream import memcpy_htod_double
from .src.cuda_stream import memcpy_htod_complex
from .src.cuda_stream import memcpy_htod_bool

# copy from device to host
from .src.cuda_stream import memcpy_dtoh_short
from .src.cuda_stream import memcpy_dtoh_int
from .src.cuda_stream import memcpy_dtoh_long
from .src.cuda_stream import memcpy_dtoh_float
from .src.cuda_stream import memcpy_dtoh_double
from .src.cuda_stream import memcpy_dtoh_complex
from .src.cuda_stream import memcpy_dtoh_bool

# getters(index)
from .src.cuda_stream import get_value_short
from .src.cuda_stream import get_value_int
from .src.cuda_stream import get_value_long
from .src.cuda_stream import get_value_float
from .src.cuda_stream import get_value_double
from .src.cuda_stream import get_value_complex
from .src.cuda_stream import get_value_bool

# getter(slice)
from .src.cuda_stream import get_slice_short
from .src.cuda_stream import get_slice_int
from .src.cuda_stream import get_slice_long
from .src.cuda_stream import get_slice_float
from .src.cuda_stream import get_slice_double
from .src.cuda_stream import get_slice_complex
from .src.cuda_stream import get_slice_bool

# setters(index)
from .src.cuda_stream import set_value_short
from .src.cuda_stream import set_value_int
from .src.cuda_stream import set_value_long
from .src.cuda_stream import set_value_float
from .src.cuda_stream import set_value_double
from .src.cuda_stream import set_value_complex
from .src.cuda_stream import set_value_bool

from .src.cuda_stream import add_short
from .src.cuda_stream import add_int
from .src.cuda_stream import add_long
from .src.cuda_stream import add_float
from .src.cuda_stream import add_double
from .src.cuda_stream import add_complex

# freebie
from .src.cuda_stream import free

# count number of CUDA-device
from .src.cuda_stream import count_device
from .src.cuda_stream import select_device


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