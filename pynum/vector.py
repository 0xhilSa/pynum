from __future__ import annotations
from typing import List,Any,Type,Tuple,Union
from . import dtypes

class Shape:
  def __init__(self, object):
    self.__length = (len(object),)
  def __repr__(self): return str(self.__length)

class Vector:
  def __init__(self, array1D: List[Type], dtype:Type=None, grad:bool=False, device:str="cpu", const:bool=False):
    self.__array1D,self.__dtype,self.__device,self.__const = self.__verify__(array1D,dtype,device,const)
    self.__shape = Shape(array1D)
    self.__grad = grad

  def __verify__(self, array1D, dtype, device, const):
    if not isinstance(array1D,list): raise TypeError("given object must be a list")
    if dtype is None:
      if any(isinstance(value, (complex,dtypes.complex128)) for value in array1D): dtype = dtypes.complex128
      elif all(isinstance(value, (int,dtypes.int64)) for value in array1D): dtype = dtypes.int64
      else:
        for index,value in enumerate(array1D): array1D[index] = float(array1D)
        dtype = dtypes.float64
    for index,value in enumerate(array1D): array1D[index] = dtype(value)
    return array1D,dtype,device,const

  def __repr__(self): return f"<Vector <dtype={self.__dtype.__name__}, shape={self.__shape}, device={self.__device}, grad={self.__grad}>>"

  @property
  def dtype(self): return self.__dtype.__name__
  @property
  def device(self): return self.__device
  @property
  def is_const(self): return self.__const

  def __getitem__(self, index:int): return self.__array1D[index]
  def to_list(self): return self.__array1D
