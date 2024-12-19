"""
vector class is only for 1D array
and doesn't support `np.ndarray`
"""

from __future__ import annotations
from typing import List,Union,Type,Optional
import numpy as np
from numba import cuda

np.int32

class Vector:
  def __init__(self, array1D, dtype:Type=float, grad:bool=False, device:str="cuda"):
    if not isinstance(array1D,list): raise TypeError("Input must be an array")
    if any(isinstance(x,list) for x in array1D): raise TypeError("Input arrray must be one-dimensional")
    if dtype:
      try:
        array1D = [dtype(x) for x in array1D]
      except ValueError as error: raise ValueError(f"Error converting elements to {dtype} : {error}")
    self.__grad = grad
    self.__length = len(array1D)
    self.__dtype = dtype
    self.__device = device.lower()
    if self.__device == "cuda":
      if not cuda.is_available(): raise RuntimeError("CUDA is not available on this system")
      self.__vector = cuda.to_device(array1D)
    elif self.__device == "cpu": self.__vector = array1D
    else: ValueError(f"Unsupported device '{device}'. Use 'cuda' or 'cpu'")

  def __repr__(self) -> str: return f"<'Vector' object at {hex(id(self))} length={self.__length} dtype={self.__dtype.__name__} grad={self.__grad} device={self.__device}>"

  def astype(self, dtype:Type) -> None:
    self.__vector[:] = [dtype(x) for x in self.__vector]
    self.__dtype = dtype

  def __getitem__(self, index:Union[int,slice]) -> Vector:
    if isinstance(index,int): return self.__vector[index]
    elif isinstance(index,slice): return self.__vector[index]

  @property
  def dtype(self) -> str: return self.__dtype.__name__

  @property
  def is_grad(self) -> bool: return self.__grad

  @property
  def device(self) -> str: return self.__device

  @property
  def length(self) -> int: return self.__length
