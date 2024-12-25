from __future__ import annotations
from typing import List, Any, Type, Tuple
from dataclasses import dataclass
from . import dtypes

@dataclass(frozen = True)
class Shape:
  dims:Tuple[int,...]
  def __repr__(self): return f"Shape{self.dims}"
  def __len__(self): return self.dims[0] if self.dims else 0

SUPPORTED_DEVICES = {"cpu"}

class Vector:
  def __init__(self, object:List[Any], dtype:Type=None, const:bool=False, device:str="cpu"):
    if device not in SUPPORTED_DEVICES: raise RuntimeError(f"Unsupported device '{device}'. Supported devices: {', '.join(SUPPORTED_DEVICES)}")
    if not object: raise ValueError("The input list cannot be empty")
    self.__object, self.__dtype  = self.__check__(object, dtype)
    self.__device = device
    self.__shape = Shape((len(object),))
    self.__const = const

  def __check__(self, object, dtype):
    if not isinstance(object, list): raise TypeError("given dtype must be a list")
    if dtype is not None: return [dtype(x) for x in object],dtype
    if any(isinstance(x,(complex,dtypes.complex128)) for x in object): dtype = dtypes.complex128
    elif all(isinstance(x,(int,dtypes.int8,dtypes.int16,dtypes.int32,dtypes.int64))for x in object): dtype = dtypes.int64
    else: dtype = dtypes.float64
    return [dtype(x) for x in object],dtype

  def __repr__(self): return f"<Vector(size={self.__shape}, dtype={self.__dtype.__name__}, device={self.__device}, device={self.__device})>"
  def __getitem__(self, index): return self.__object[index]
  def __setitem__(self, index, value):
    if self.__const: raise RuntimeError("can't modify a constant vector")
    self.__object[index] = value

  @property
  def dtype(self): return self.__dtype.__name__
  @property
  def device(self): return self.__device
  @property
  def shape(self): return self.__shape
  
  def is_const(self): return self.__const
