from __future__ import annotations
from typing import List, Any, Type, Tuple, Optional, Union
import numpy as np
from .cuda import *
"""
from .src.cuda_stream import (
  cuda_alloc_int, cuda_alloc_long, cuda_alloc_double, cuda_alloc_complex,
  memcpy_htod_int, memcpy_htod_long, memcpy_htod_double, memcpy_htod_complex,
  memcpy_dtoh_int, memcpy_dtoh_long, memcpy_dtoh_double, memcpy_dtoh_complex,
  get_value_int, get_value_long, get_value_double, get_value_complex,
  get_slice_int, get_slice_long, get_slice_double, get_slice_complex,
  set_value_int, set_value_long, set_value_double, set_value_complex,
  set_slice_int, set_slice_long,
  cuda_free, cuda_query_free_memory
)"""
from .ops import Ops, GroupOp, ops_mapping


DEVICE_SUPPORTED = ["CPU","CUDA"]
FLOATING = [float, np.float16, np.float32, np.float64, np.float128]
INTEGER = [int, np.int8, np.int16, np.int32, np.int64]
COMPLEX = [complex, np.complex128]
ALL = FLOATING + INTEGER + COMPLEX


class Vector:
  @staticmethod
  def cuda_alloc(obj:List[Any], dtype:Type):
    if dtype in FLOATING: return cuda_alloc_double(obj)
    elif dtype in INTEGER: return cuda_alloc_long(obj)
    elif dtype in COMPLEX: return cuda_alloc_complex(obj)
  @staticmethod
  def memcpy_htod(ptr, obj:List[Any], dtype:Type):
    if dtype in FLOATING: return memcpy_htod_double(ptr, obj)
    elif dtype in INTEGER: return memcpy_htod_long(ptr, obj)
    elif dtype in COMPLEX: return memcpy_htod_complex(ptr, obj)
  @staticmethod
  def memcpy_dtoh(ptr, length:int, dtype:Type):
    if dtype in FLOATING: return memcpy_dtoh_double(ptr, length)
    elif dtype in INTEGER: return memcpy_dtoh_long(ptr, length)
    elif dtype in COMPLEX: return memcpy_dtoh_complex(ptr, length)
  @staticmethod
  def get_value(ptr, index:int, dtype:Type):
    if dtype in FLOATING: return get_value_double(ptr, index)
    elif dtype in INTEGER: return get_value_long(ptr, index)
    elif dtype in COMPLEX: return get_value_complex(ptr, index)
  @staticmethod
  def get_slice_value(ptr, start:int, stop:int, step:int, dtype:Type):
    if dtype in FLOATING: return get_slice_double(ptr, start, stop, step)
    elif dtype in INTEGER: return get_slice_long(ptr, start, stop, step)
    elif dtype in COMPLEX: return get_slice_complex(ptr, start, stop, step)
  @staticmethod
  def set_value(ptr, index:int, value, dtype:Type):
    if dtype in FLOATING: set_value_double(ptr, index, value)
    if dtype in INTEGER: set_value_long(ptr, index, value)
    if dtype in COMPLEX: set_value_complex(ptr, index, value)
  @staticmethod
  def set_slice_value(ptr, start:int, stop:int, step:int, value, dtype:Type):
    if dtype is INTEGER: set_slice_long(ptr, start, stop, step, value)
  @staticmethod
  def free(ptr): cuda_free(ptr)
  def __init__(
    self,
    obj:List[Union[int,float,complex,np.integer,np.floating,np.complex128]],
    dtype:Type=None,
    const:bool=False,
    device:str="CPU",
    requires_grad:bool=False
  ) -> None:
    self.__obj, self.__dtype, self.__device, self.__length = self.__check__(obj, dtype, device)
    self.__const = const
    self.__grad = requires_grad
  def __check__(self, obj:List[Union[int,float,complex,np.integer,np.floating,np.complex128]], dtype:Type, device:str) -> Tuple[Any, Type, str, int]:
    if dtype is None:
      if any(isinstance(x,(complex,np.complex128)) for x in obj): dtype = np.complex128
      elif all(isinstance(x,(int,np.integer)) for x in obj): dtype = np.int64
      elif any(isinstance(x,(float,np.floating)) for x in obj): dtype = np.float64
      else: raise ValueError("Mixed or unsupported types in obj. Provide a specific")
    obj = [dtype(x) for x in obj]
    device = device.upper()
    if device not in DEVICE_SUPPORTED: raise ValueError(f"Unsupported device '{device}', it could be either 'CPU' or 'CUDA'")
    if device == "CUDA":
      try:
        CUDA_ptr = Vector.cuda_alloc(obj, dtype)
        Vector.memcpy_htod(CUDA_ptr, obj, dtype)
        return CUDA_ptr, dtype, device, len(obj)
      except Exception as e:
        cuda_free(CUDA_ptr)
        raise RuntimeError(f"CUDA memory initialization failed: {e}") 
    return obj, dtype, device, len(obj)
  def __repr__(self) -> str: return f"<Vector(length={self.__length}, dtype={self.__dtype.__name__}, device={self.__device}, const={self.__const})>"
  @property
  def device(self) -> str: return self.__device
  @property
  def dtype(self) -> Type: return self.__dtype.__name__
  @property
  def length(self) -> int: return self.__length
  def numpy(self):
    if self.__device == "CUDA": return np.array(Vector.memcpy_dtoh(self.__obj, self.__length, self.__dtype))
    return np.array(self.__obj)
  def __del__(self):
    if self.__device == "CUDA": cuda_free(self.__obj)
  def __getitem__(self, index:Union[int,slice]):
    if isinstance(index,int):
      if index < 0: index += self.__length
      if not (0 <= index < self.__length): raise IndexError("Index out of range")
      if self.__device == "CUDA": return Vector.get_value(self.__obj, index, self.__dtype)
      else: return self.__dtype(self.__obj[index])
    elif isinstance(index,slice):
      start, stop, step = index.start, index.stop, index.step
      step = step or 1
      if step == 0: raise ValueError("Slice step must be greater than 0")
      if start is None: start = 0 if step > 0 else self.__length - 1
      if stop is None: stop = self.__length if step > 0 else -1
      if start < 0: start += self.__length
      if stop < 0: stop += self.__length
      if self.__device == "CUDA": return Vector(Vector.get_slice_value(self.__obj, start, stop, step, self.__dtype), dtype=self.__dtype, device="CUDA")
      elif self.__device == "CPU": return Vector(self.__obj[index], dtype=self.__dtype, device=self.__device)
    else: raise TypeError("Invalid index type. Must be an int or slice")
  def __setitem__(self, index:Union[int,slice], value:Any):
    if self.__const: raise RuntimeError("cannot make changes on immutable Vector")
    if isinstance(index,int):
      if index < 0: index += self.__length
      if not (0 <= index < self.__length): raise IndexError("Index out of range")
      if self.__device == "CUDA": Vector.set_value(self.__obj, index, value, self.__dtype)
      elif self.__device == "CPU": self.__obj[index] = value
    elif isinstance(index, slice):
      start, stop, step = index.start, index.stop, index.step
      step = step or 1
      start = start if start is not None else (0 if step > 0 else self.__length - 1)
      stop = stop if stop is not None else (self.__length if step > 0 else -1)
      start = start + self.__length if start < 0 else start
      stop = stop + self.__length if stop < 0 else stop
      if not isinstance(value, (list, tuple, np.ndarray)): raise TypeError("Assigned value must be a list, tuple, or numpy array.")
      slice_length = len(range(start, stop, step))
      if len(value) != slice_length: raise ValueError(f"Length of value ({len(value)}) does not match the slice length ({slice_length}).")
      indices = range(start, stop, step)
      if self.__device == "CUDA":
          for idx, val in zip(indices, value): Vector.set_value(self.__obj, idx, val, self.__dtype)
      elif self.__device == "CPU":
          for idx, val in zip(indices, value): self.__obj[idx] = val
    else: raise TypeError("Invalid index type. Must be an int or slice")
  def to(self, device:str):
    if device.upper() not in DEVICE_SUPPORTED: raise ValueError(f"Unsupported device '{device}'")
    if device.upper() == self.__device: return
    if device.upper() == "CUDA":
      CUDA_ptr = Vector.cuda_alloc(self.__obj, self.__dtype)
      Vector.memcpy_htod(CUDA_ptr, self.__obj, self.__dtype)
      self.__obj = CUDA_ptr
    elif device.upper() == "CPU": self.__obj = Vector.memcpy_dtoh(self.__obj, self.__length, self.__dtype)
    self.__device = device.upper()
  def CUDA(self):
    if self.__device == "CUDA": return
    CUDA_ptr = Vector.cuda_alloc(self.__obj, self.__dtype)
    Vector.memcpy_htod(CUDA_ptr, self.__obj, self.__dtype)
    self.__obj = CUDA_ptr
    self.__device = "CUDA"
  def CPU(self):
    if self.__device == "CPU": return
    self.__obj = Vector.memcpy_dtoh(self.__obj, self.__length, self.__dtype)
    self.__device = "CPU"
  def __add__(self, other): pass
