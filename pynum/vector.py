from __future__ import annotations
from typing import List, Any, Type, Tuple, Optional, Union
import numpy as np
from .cuda import *
from .ops import Ops, GroupOp, ops_mapping


FLOATING = [float, np.float16, np.float32, np.float64, np.float128]
INTEGER = [int, np.int8, np.int16, np.int32, np.int64]
COMPLEX = [complex, np.complex128]
BOOLEAN = [bool]
ALL = BOOLEAN + FLOATING + INTEGER + COMPLEX
DEVICE_SUPPORTED = ["CPU", "CUDA"]


class Vector:
  def __init__(self, array:List[Type], dtype:Type=None, device:str="CPU", const:bool=False):
    self.__array, self.__length, self.__dtype, self.__device = Vector.__check__(array, dtype, device)
    self.__const = const
  @staticmethod
  def __check__(array:List[Type], dtype:Type, device:str):
    if dtype is None:
      if any(isinstance(x,(complex, np.complex128)) for x in array): dtype = complex
      elif all(isinstance(x,(int, np.integer)) for x in array): dtype = int
      elif any(isinstance(x,(float,np.floating)) for x in array): dtype = float
      elif all(isinstance(x,bool) for x in array): dtype = bool
      else: raise TypeError(f"An invalid dtype: '{dtype.__name__}' were given")
    elif dtype not in ALL: raise TypeError("An invalid dtype: '{dtype.__name__}' were given")
    array = [dtype(x) for x in array]
    length = len(array)
    if device.upper() == "CPU": pass
    elif device.upper() == "CUDA":
      if dtype == int:
        ptr = alloc_long(array)
        memcpy_htod_long(ptr, array)
      elif dtype == float:
        ptr = alloc_double(array)
        memcpy_htod_double(ptr, array)
      elif dtype == complex:
        ptr = alloc_complex(array)
        memcpy_htod_complex(ptr, array)
      elif dtype == bool: raise NotImplementedError("CUDA allocation is not implemented for boolean dtype")
      array = ptr
    else: raise RuntimeError(f"Device: '{device}' isn't support/available on this system")
    return array, length, dtype, device.upper()
  def __repr__(self): return f"<Vector(length={self.__length}, dtype={self.__dtype.__name__}, device={self.__device}, const={self.__const})>"
  def __len__(self): return self.__length
  def __del__(self):
    if self.__device == "CUDA": free(self.__array)
    del self
  def __getitem__(self, index:Union[int,slice]):
    if self.__device == "CUDA":
      if isinstance(index,int):
        if self.__dtype in FLOATING: return get_value_double(self.__array, index)
        elif self.__dtype in INTEGER: return get_value_long(self.__array, index)
        elif self.__dtype in COMPLEX: return get_value_complex(self.__array, index)
      elif isinstance(index,slice):
        start, stop, step = index.start, index.stop, index.step
        step = step or 1
        if step == 0: raise ValueError("Slice step must be greater than 0")
        if start is None: start = 0 if step > 0 else self.__length - 1
        if stop is None: stop = self.__length if step > 0 else -1
        if start < 0: start += self.__length
        if stop < 0: stop += self.__length
        if self.__dtype in FLOATING: return Vector(get_slice_double(self.__array, start, stop, step), dtype=self.__dtype, device="CUDA")
        elif self.__dtype in INTEGER: return Vector(get_slice_long(self.__array, start, stop, step), dtype=self.__dtype, device="CUDA")
        elif self.__dtype in COMPLEX: return Vector(get_slice_complex(self.__array, start, stop, step), dtype=self.__dtype, device="CUDA")
      else: raise ValueError("Given index must be either an integer or a slice")
    return Vector(self.__array[index], dtype=self.__dtype, device="CPU")
  def raw(self): return self.__array    # this is for testing
  def cuda(self):
    if self.__device == "CPU":
      if self.__dtype in FLOATING:
        ptr = alloc_double(self.__array)
        memcpy_htod_double(ptr, self.__array)
      elif self.__dtype in INTEGER:
        ptr = alloc_long(self.__array)
        memcpy_htod_long(ptr, self.__array)
      elif self.__dtype in COMPLEX:
        ptr = alloc_complex(self.__array)
        memcpy_htod_complex(ptr, self.__array)
      elif self.__dtype in BOOLEAN: raise NotImplementedError("CUDA allocation is not implemented for boolean dtype")     # unreachable
      self.__array = ptr
      self.__device = "CUDA"
  def cpu(self):
    if self.__device == "CUDA":
      if self.__dtype in FLOATING: y = memcpy_dtoh_double(self.__array, self.__length)
      elif self.__dtype in INTEGER: y = memcpy_dtoh_long(self.__array, self.__length)
      elif self.__dtype in COMPLEX: y = memcpy_dtoh_complex(self.__array, self.__length)
      elif self.__dtype in BOOLEAN: raise NotImplementedError("CUDA deallocation is not implemented for boolean dtype")   # unreachable
      free(self.__array)
      self.__array = y
      self.__device = "CPU"
  def to(self, device:str):
    if device.upper() == "CUDA": self.cuda()
    elif device.upper() == "CPU": self.cpu()
  def numpy(self):
    if self.__device == "CUDA":
      if self.__dtype in FLOATING: return np.array(memcpy_dtoh_double(self.__array, self.__length))
      elif self.__dtype in INTEGER: return np.array(memcpy_dtoh_long(self.__array, self.__length))
      elif self.__dtype in COMPLEX: return np.array(memcpy_dtoh_complex(self.__array, self.__length))
    return np.array(self.__array)