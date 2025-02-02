from __future__ import annotations
from typing import List, Type, Union
import numpy as np
import random
from graphviz import Digraph

# pre-defined
from .csrc import host, pycu
from .exceptions import *
from .dtype import *


class Vector:
  def __init__(self, array, dtype:Union[Type,DType], device:str="cpu", const:bool=False):
    self.__pointer, self.__length, self.__dtype, self.__device = Vector.__check__(array, dtype, device)
    self.__const = const
  def __repr__(self): return f"<Vector(length={self.__length}, dtype='{self.__dtype.name}', device={self.__device}, const={self.__const})>"
  @staticmethod
  def __check__(array:List, dtype:DType, device:str):
    # if user provide builtin dtypes
    if dtype == int: dtype = int64
    elif dtype == float: dtype = float64
    elif dtype == complex: dtype = complex128
    elif dtype == bool: dtype = bool_
    length = len(array)
    if device.lower() == "cpu": array = host.array(array, dtype.fmt)
    elif device.lower() == "cuda": array = pycu.toCuda(host.array(array, dtype.fmt), length, dtype.fmt)
    return array, length, dtype, device
  @property
  def dtype(self): return self.__dtype.name
  @property
  def fmt(self): return self.__dtype.fmt
  @property
  def device(self): return self.__device
  def pointer(self): return self.__pointer  # this is for testing and will later remove this func
  def is_const(self): return self.__const
  def list(self):
    if self.__device == "cuda": return host.toList(pycu.toHost(self.__pointer, self.__length, self.__dtype.fmt), self.__length, self.__dtype.fmt)
    return host.toList(self.__pointer, self.__length, self.__dtype.fmt)
  def numpy(self): return np.array(self.list())
  def astype(self, dtype:Union[Type,DType]):
    if dtype == int: dtype = int64
    elif dtype == float: dtype = float64
    elif dtype == complex: dtype = complex128
    elif dtype == bool: dtype = bool_
  def __getitem__(self, index:int):
    if not (0 <= index < self.__length): raise IndexError(f"Index: {index} is out of bound!")
    if self.__device == "cpu": return Vector([host.get(self.__pointer, index, self.fmt)], dtype=self.__dtype)
    elif self.__device == "cuda": return Vector(host.toList(pycu.toHost(pycu.get(self.__pointer, index, self.__dtype.fmt), 1, self.__dtype.fmt), 1, self.__dtype.fmt), dtype=self.__dtype, device=self.__device)
  def __setitem__(self, index:int, value):
    pass