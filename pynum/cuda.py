# allocators for different data types
from .src.cuda_stream import cuda_alloc_int
from .src.cuda_stream import cuda_alloc_long
from .src.cuda_stream import cuda_alloc_double
from .src.cuda_stream import cuda_alloc_complex

# copy from host to device
from .src.cuda_stream import memcpy_htod_int
from .src.cuda_stream import memcpy_htod_long
from .src.cuda_stream import memcpy_htod_double
from .src.cuda_stream import memcpy_htod_complex

# copy from device to host
from .src.cuda_stream import memcpy_dtoh_int
from .src.cuda_stream import memcpy_dtoh_long
from .src.cuda_stream import memcpy_dtoh_double
from .src.cuda_stream import memcpy_dtoh_complex

# getters(index)
from .src.cuda_stream import get_value_int
from .src.cuda_stream import get_value_long
from .src.cuda_stream import get_value_double
from .src.cuda_stream import get_value_complex

# getter(slice)
from .src.cuda_stream import get_slice_int
from .src.cuda_stream import get_slice_long
from .src.cuda_stream import get_slice_double
from .src.cuda_stream import get_slice_complex

# setters
from .src.cuda_stream import set_value_int
from .src.cuda_stream import set_value_long
from .src.cuda_stream import set_value_double
from .src.cuda_stream import set_value_complex

from .src.cuda_stream import set_slice_int
from .src.cuda_stream import set_slice_long
#from .src.cuda_stream import set_slice_double
#from .src.cuda_stream import set_slice_complex

# free-bies
from .src.cuda_stream import cuda_free
from .src.cuda_stream import cuda_query_free_memory

import subprocess

def is_cuda_available():
  try:
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0
  except FileNotFoundError: pass
  try:
    result = subprocess.run(["nvcc","--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0
  except FileNotFoundError: pass
  return False
