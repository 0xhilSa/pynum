from __future__ import annotations
from typing import Any, List, Type, Union
import numpy as np
from .dtype import *
from .csrc import host, pycu


class Vector:
  @staticmethod
  def __from_builtin2custom(dtype:Type) -> DType:
    if dtype not in BUILTIN: raise TypeError(f"Invalid DType: '{dtype}'")
    if dtype == int: return int32
    elif dtype == float: return float32
    elif dtype == complex: return complex128
    elif dtype == bool: return boolean
    raise TypeError(f"'{dtype}' is incompatible")
  @staticmethod
  def __from_custom2builtin(dtype:DType) -> Type:
    if dtype in INTEGER: return int
    elif dtype in FLOATING: return float
    elif dtype in COMPLEX: return complex
    elif dtype in BOOLEAN: return bool
    raise TypeError(f"'{dtype}' is incompatible")
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
  def __repr__(self): return f"<Vector(length={self.__length}, dtype='{self.__dtype.get_name()}', device='{self.__device}', const={self.__const})>"
  def __len__(self): return self.__length
  @property
  def ptr(self): return self.__pointer
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
  def __setitem__(self, index:Union[int,slice], value:Union[List[Any],Any]):
    if isinstance(index, int):
      value = Vector.__from_custom2builtin(self.__dtype)(value)
      if self.__device == "cpu": host.setitem_index(self.__pointer, value, self.__length, index, self.__dtype.get_fmt())
      elif self.__device == "cuda": pycu.setitem_index(self.__pointer, value, self.__length, index, self.__dtype.get_fmt())
    elif isinstance(index, slice):
      start, stop, step = index.start, index.stop, index.step
      if step is None: step = 1
      dtype = Vector.__from_custom2builtin(self.__dtype)
      for index, element in enumerate(value): value[index] = dtype(element)
      if self.__device == "cpu": host.setitem_slice(self.__pointer, value, self.__length, start, stop, step, self.__dtype.get_fmt())
      elif self.__device == "cuda": pycu.setitem_slice(self.__pointer, value, self.__length, start, stop, step, self.__dtype.get_fmt())
  def bitwise_and(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.bitwise_and_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def bitwise_or(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.bitwise_or_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def bitwise_not(self):
    if self.__device == "cpu": return Vector(host.toList(host.bitwise_not_vector(self.__pointer, self.__length, self.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise RuntimeError(f"Invalid Device Provided")
  def bitwise_xor(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.bitwise_xor_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def bitwise_nand(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return self.bitwise_and(other).bitwise_not()
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def bitwise_nor(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return self.bitwise_or(other).bitwise_not()
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def bitwise_xnor(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return self.bitwise_xor(other).bitwise_not()
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def add(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.add_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def sub(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.sub_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def mul(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.mul_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def tdiv(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.div_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def pow(self, other:Vector, reverse:bool=False):
    if reverse: self, other = other, self
    if self.__device == "cpu": return Vector(host.toList(host.pow_vector(self.__pointer, other.__pointer, self.__length, other.__length, self.__dtype.get_fmt(), other.__dtype.get_fmt()), self.__length, self.__dtype.get_fmt()), self.__dtype)
    elif self.__device == "cuda": raise NotImplementedError
    raise TypeError(f"Invalid dtype provided: {type(other).__name__}")
  def __add__(self, other:Vector): return self.add(other)
  def __sub__(self, other:Vector): return self.sub(other)
  def __mul__(self, other:Vector): return self.mul(other)
  def __truediv__(self, other:Vector): return self.tdiv(other)
  def __pow__(self, other:Vector): return self.pow(other)
  def __radd__(self, other:Vector): return self.add(other, True)
  def __rsub__(self, other:Vector): return self.sub(other, True)
  def __rmul__(self, other:Vector): return self.mul(other, True)
  def __rtruediv__(self, other:Vector): return self.tdiv(other, True)
  def __rpow__(self, other:Vector): return self.pow(other, True)
  def __and__(self, other:Vector): return self.bitwise_and(other, False)
  def __or__(self, other:Vector): return self.bitwise_or(other, False)
  def __invert__(self): return self.bitwise_not()
  def __xor__(self, other:Vector): return self.bitwise_xor(other, False)
  def __rand__(self, other:Vector): return self.bitwise_and(other, True)
  def __ror__(self, other:Vector): return self.bitwise_or(other, True)
  def __rxor__(self, other:Vector): return self.bitwise_xor(other, True)
