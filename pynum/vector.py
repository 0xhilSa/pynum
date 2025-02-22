from __future__ import annotations
from typing import Any, List, Type, Union
from .dtype import *


class Vector:
  def __init__(self, array:List[Any], dtype:Union[Type,DType], device:str="cpu", const:bool=False):
    self.__pointer, self.__length, self.__dtype = Vector.__check__(array, dtype, device)

  @staticmethod
  def __check__(array:List[Any], dtype:Union[Type,DType], device:str): pass
