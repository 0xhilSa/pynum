from __future__ import annotations
from typing import Any, List, Type, Union
import numpy as np
from .dtype import *
from .csrc import host, pycu



class Vector:
  @staticmethod
  def __from_builtin2custom(dtype:Type):
    if dtype not in BUILTIN: raise TypeError(f"Invalid DType: '{dtype}'")
    if dtype == int: return int64
    elif dtype == float: return float64
    elif dtype == complex: return complex256
    return boolean
  def __init__(self, array:List[Any], dtype:Union[Type,DType], device:str="cpu", const:bool=False):
    self.__pointer, self.__length, self.__dtype, self.__device = Vector.__check__(array, dtype, device)
    self.__const = const
  @staticmethod
  def __check__(array:List[Any], dtype:Union[Type,DType], device:str):
    if not isinstance(array, list): raise TypeError("Expected a list for array")
    if any(isinstance(el, list) for el in array): raise ValueError("Only 1D lists are allowed. Nested lists are not supported")
    if len(array) == 0: raise ValueError("Array cannot be empty")
    if dtype in BUILTIN: dtype = Vector.__from_builtin2custom(dtype)
    elif not isinstance(dtype, DType): raise TypeError(f"Invalid dtype {dtype}. Must be int, float, complex, bool, or a custom DType")
    fmt = dtype.get_fmt()
    length = len(array)
    array = host.array(array, fmt)
    if device.lower() == "cuda": array = pycu.toCuda(array, length, fmt)
    return array, length, dtype, device.lower()
  def __repr__(self): return f"<Vector(length={self.__length}, dtype='{self.__dtype.get_name()}', device='{self.__device}')>"
  @property
  def device(self): return self.__device
  @property
  def dtype(self): return self.__dtype
  @property
  def fmt(self): return self.__dtype.get_fmt()
  def pointer(self): return self.__pointer    # for testing
  def is_const(self): return self.__const
  def list_(self):
    if self.__device == "cuda": return host.toList((pycu.toHost(self.__pointer, self.__length, self.__dtype.get_fmt())), self.__length, self.__dtype.get_fmt())
    return host.toList(self.__pointer, self.__length, self.__dtype.get_fmt())
  def numpy(self):
    if self.__device == "cuda":
      x = pycu.toHost(self.__pointer, self.__length, self.__dtype.fmt)
      return np.array(host.toList(x, self.__length, self.__dtype.fmt))
    return np.array(self.list_())
  def cuda(self):
    if self.__device == "cpu": return Vector(self.list_(), self.__dtype, device="cuda")
    return self
  def cpu(self):
    if self.__device == "cuda": return Vector(self.list_(), self.__dtype, device="cpu")
    return self
  def __getitem__(self, index:Union[int,slice]):
    if isinstance(index,int):
      if self.__device == "cuda": res = pycu.toHost(pycu.getitem_index(self.__pointer, index, self.__dtype.get_fmt()), 1, self.__dtype.get_fmt())
      elif self.__device == "cpu": res = host.getitem_index(self.__pointer, self.__length, index, self.__dtype.get_fmt())
      else: raise NotImplementedError
      res = host.toList(res, 1, self.__dtype.get_fmt())
      return Vector(res, dtype=self.__dtype, device=self.__device)
    elif isinstance(index, slice):
      start, stop, step = index.start, index.stop, index.step
      if start is None: start = 0
      if stop is None: stop = self.__length
      if step is None: step = 1
      sliced_len = max(0, (stop - start + step - (1 if step > 0 else -1)) // step)
      if self.__device == "cuda": res = pycu.toHost(pycu.getitem_slice(self.__pointer, start, stop, step, self.__dtype.get_fmt()), sliced_len, self.__dtype.get_fmt())
      elif self.__device == "cpu": res = host.getitem_slice(self.__pointer, self.__length, start, stop, step, self.__dtype.get_fmt())
      else: raise NotImplementedError
      res = host.toList(res, sliced_len, self.__dtype.get_fmt())
      return Vector(res, self.__dtype, self.__device)
