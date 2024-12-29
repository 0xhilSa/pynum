from __future__ import annotations
from typing import List, Any, Type, Tuple, Optional, Union
import random
import numpy as np
from .ops import Ops, GroupOp, ops_mapping
from .device import Device, CUDAManager
from .shape import Shape


cuda_manager = CUDAManager()

class Vector:
  def __init__(self, object:List[Any], dtype:Type=None, const:bool=False, device:str="cpu"):
    if device.upper() not in Device.SUPPORTED_DEVICES: raise RuntimeError(f"Unsupported device '{device}'. Supported devices: {', '.join(Device.SUPPORTED_DEVICES)}")
    if not object: raise ValueError("The input list cannot be empty")
    self.__object, self.__dtype = self.__check__(object, dtype)
    self.__device = Device(device.upper())
    self.__shape = Shape((len(object),))
    self.__const = const
    #self.__cuda_handle = None
    #if self.__device.to_str() == "CUDA": self.__cuda_handle = cuda_manager.allocate(self.__object, self.__dtype)
  def __check__(self, object, dtype):
    if not isinstance(object, list): raise TypeError("Input must be a list")
    if dtype is not None: return [dtype(x) for x in object], dtype
    if any(isinstance(x, (complex, np.complex128)) for x in object): dtype = np.complex128
    elif all(isinstance(x, int) for x in object): dtype = np.int64
    else: dtype = np.float64
    return [dtype(x) for x in object], dtype
  def __repr__(self): return f"<Vector(size={self.__shape}, dtype={self.__dtype.__name__}, const={self.__const}, device={self.__device})>"
  def __getitem__(self, index: Union[int, slice]):
    if isinstance(index, slice):
       sliced_obj = self.__object[index]
       return Vector(sliced_obj, dtype=self.__dtype, const=self.__const, device=self.__device)
    return self.__object[index]
  def __setitem__(self, index, value):
    if self.__const: raise RuntimeError("Can't modify a constant vector")
    self.__object[index] = value
  @property
  def dtype(self): return self.__dtype.__name__
  @property
  def device(self): return self.__device
  @property
  def shape(self): return self.__shape
  def is_const(self): return self.__const
  @staticmethod
  def rand(length:int, dtype:Type=np.float64, lower_bound:int=0, upper_bound:int=1, seed:Optional[int]=None, const:bool=False, device="cpu"):
    if seed is not None: random.seed(seed)
    if dtype in (float, np.float32, np.float64): rand_vec = [random.uniform(lower_bound, upper_bound) for _ in range(length)]
    elif dtype in (int, np.int8, np.int16, np.int32, np.int64): rand_vec = [random.randint(lower_bound, upper_bound) for _ in range(length)]
    else: raise TypeError(f"Random generation not supported for dtype {dtype}")
    return Vector(rand_vec, dtype=dtype, const=const, device=device)
  @staticmethod
  def zeros(length:int, dtype:Type=np.float64, const:bool=False, device="cpu"): return Vector([0] * length, dtype=dtype, const=const, device=device)
  @staticmethod
  def ones(length:int, dtype:Type=np.float64, const:bool=False, device="cpu"): return Vector([1] * length, dtype=dtype, const=const, device=device)
  @staticmethod
  def fill(value:Any, length:int, dtype:Type=np.float64, const: bool = False, device="cpu"): return Vector([value] * length, dtype=dtype, const=const, device=device)
  def add(self, x:Union[Vector,Any], ops:Ops, const:bool=False, device:str="cpu"): pass
