from __future__ import annotations
from typing import List, Type, Union
import numpy as np
import random
from graphviz import Digraph

# declared
from .csrc import host
from .exceptions import *
from .dtype import *


class Vector:
  def __init__(self, array, dtype:Union[Type,DType], device:str="CPU", const:bool=False):
    self.__pointer, self.__length, self.__dtype, self.__fmt, self.__device = Vector.__check__(array, dtype, device)
    self.__const = const
  def __repr__(self): return f"<Vector(length={self.__length}, dtype='{self.__dtype}', device={self.__device}, const={self.__const})>"
  @staticmethod
  def __check__(array, dtype, device):
    # if user provide builtin dtypes
    if dtype == int: dtype = int64
    elif dtype == float: dtype = float64
    elif dtype == complex: dtype = complex128
    length = len(array)
    if device.upper() == "CPU": array = host.array(array, dtype.fmt)
    elif device.upper() == "CUDA": raise NotImplementedError
    return array, length, dtype.name, dtype.fmt, device
  @property
  def dtype(self): return self.__dtype
  @property
  def fmt(self): return self.__fmt
  @property
  def device(self): return self.__device
  def is_const(self): return self.__const
  def list(self): return host.toList(self.__pointer, self.__fmt, self.__length)
  def numpy(self): return np.array(self.list())
