from __future__ import annotations
from typing import List, Any, Type, Tuple, Optional, Union
import random
import numpy as np
from .src.cu_manager import (
  cuda_alloc_int, cuda_alloc_long, cuda_alloc_double, cuda_alloc_complex,
  memcpy_htod_int, memcpy_htod_long, memcpy_htod_double, memcpy_htod_complex,
  memcpy_dtoh_int, memcpy_dtoh_long, memcpy_dtoh_double, memcpy_dtoh_complex,
  cuda_free, cuda_query_free_memory
)
from .ops import Ops, GroupOp, ops_mapping


DEVICE_SUPPORTED = ["cpu","cuda"]
FLOATING = [float, np.float16, np.float32, np.float64, np.float128]
INTEGER = [int, np.int8, np.int16, np.int32, np.int64]
COMPLEX = [complex, np.complex128]
ALL = FLOATING + INTEGER + COMPLEX


class Vector:
  def cuda_alloc(obj:List[Any], dtype:Type):
    if dtype in FLOATING: return cuda_alloc_double(obj)
    elif dtype in INTEGER: return cuda_alloc_long(obj)
    elif dtype in COMPLEX: return cuda_alloc_complex(obj)
  def memcpy_htod(ptr, obj:List[Any], dtype:Type):
    if dtype in FLOATING: return memcpy_htod_double(ptr, obj)
    elif dtype in INTEGER: return memcpy_htod_long(ptr, obj)
    elif dtype in COMPLEX: return memcpy_htod_complex(ptr, obj)
  def memcpy_dtoh(ptr, length:int, dtype:Type):
    if dtype in FLOATING: return memcpy_dtoh_double(ptr, length)
    elif dtype in INTEGER: return memcpy_dtoh_long(ptr, length)
    elif dtype in COMPLEX: return memcpy_dtoh_complex(ptr, length)
  def __init__(
    self,
    obj:List[Union[int,float,complex,np.integer,np.floating,np.complex128]],
    dtype:Type=None,
    const:bool=False,
    device:str="cpu"
  ) -> None:
    self.__obj, self.__dtype, self.__device, self.__length = self.__check__(obj, dtype, device)
    self.__const = const
  def __check__(self, obj:List[Union[int,float,complex,np.integer,np.floating,np.complex128]], dtype:Type, device:str) -> Tuple[Any, Type, str, int]:
    if dtype is None:
      if any(isinstance(x,(complex,np.complex128)) for x in obj): dtype = np.complex128
      elif all(isinstance(x,(int,np.integer)) for x in obj): dtype = np.int64
      elif any(isinstance(x,(float,np.floating)) for x in obj): dtype = np.float64
      else: raise ValueError("Mixed or unsupported types in obj. Provide a specific")
    obj = [dtype(x) for x in obj]
    device = device.lower()
    if device not in DEVICE_SUPPORTED: raise ValueError(f"Unsupported device '{device}', it could be either 'cpu' or 'cuda'")
    if device == "cuda":
      cuda_ptr = Vector.cuda_alloc(obj, dtype)
      Vector.memcpy_htod(cuda_ptr, obj, dtype)
      # except Exception as e:
        # cuda_free(cuda_ptr)
        # raise RuntimeError(f"CUDA memory initialization failed: {e}") 
      return cuda_ptr, dtype, device, len(obj)
    return obj, dtype, device, len(obj)
  def __repr__(self) -> str: return f"<Vector(length={self.__length}, dtype={self.__dtype.__name__}, device={self.__device}, const={self.__const})>"
  @property
  def device(self) -> str: return self.__device
  @property
  def dtype(self) -> Type: return self.__dtype.__name__
  @property
  def length(self) -> int: return self.__length
  def numpy(self):
    if self.__device == "cuda":
      # if self.__dtype in          # something needs to be done here
      return np.array(Vector.memcpy_dtoh(self.__obj, self.__length, ))
    return np.array(self.__obj)
  def __del__(self):
    if self.__device == "cuda": cuda_free(self.__obj)
  def __getitem__(self, index:int):
    if not (0 <= index < self.__length): raise IndexError(f"Index must lie from 0(included) to {self.__length - 1}(included)")
    if self.__device == "cpu": return self.__obj[index]
    elif self.__device == "cuda": return Vector.memcpy_dtoh(self.__obj, self.__length, self.__dtype)[index]
  def __setitem__(self, index:int, value:Any): pass
  def to(self, device:str):
    if device.lower() not in DEVICE_SUPPORTED: raise ValueError("Unsupported device '{device}'")
    if device.lower() == self.__device: return
    if device.lower() == "cuda":
      cuda_ptr = Vector.cuda_alloc(self.__obj, self.__dtype)
      Vector.memcpy_htod(cuda_ptr, self.__obj, self.__dtype)
      self.__obj = cuda_ptr
    elif device.lower() == "cpu": self.__obj = Vector.memcpy_dtoh(self.__obj, self.__length, self.__dtype)
    self.__device = device.lower()
  def cuda(self):
    if self.__device == "cuda": return
    cuda_ptr = Vector.cuda_alloc(self.__obj, self.__dtype)
    Vector.memcpy_htod(cuda_ptr, self.__obj, self.__dtype)
    self.__obj = cuda_ptr
    self.__device = "cuda"
  def cpu(self):
    if self.__device == "cpu": return
    self.__obj = Vector.memcpy_dtoh(self.__obj, self.__length, self.__dtype)
    self.__device = "cpu"
