from __future__ import annotations
from typing import List, Any, Type, Tuple
from . import dtypes

class Shape:
  def __init__(self, obj: Any): self.__length = (len(obj),)
  def __repr__(self): return str(self.__length)
  def __len__(self): return self.__length[0]

class Vector:
  def __init__(self, object:List[Any], dtype:Type=None, const:bool=False, device:str="cpu"):
    if device != "cpu": raise RuntimeError(f"Sorry, this library is in under construction, it doesn't supports for device={device} for now!!")
    self.__object, self.__dtype  = self.__check__(object, dtype)
    self.__device = device
    self.__shape = Shape(object)
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
