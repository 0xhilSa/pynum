"""
these functions are written in CUDA and
defined at './pynum/src/cuda_stream.cu'
"""

from typing import List, Type
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

# ops on CUDA device
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

# converters
from .src.cuda_stream import long2double
from .src.cuda_stream import long2complex
from .src.cuda_stream import long2bool
from .src.cuda_stream import double2long
from .src.cuda_stream import double2complex
from .src.cuda_stream import double2bool
from .src.cuda_stream import complex2long
from .src.cuda_stream import complex2double
from .src.cuda_stream import complex2bool
from .src.cuda_stream import bool2long
from .src.cuda_stream import bool2double
from .src.cuda_stream import bool2complex

from .src.cuda_stream import copy_long
from .src.cuda_stream import copy_double
from .src.cuda_stream import copy_complex
from .src.cuda_stream import copy_bool

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