import os,sys,subprocess
from dataclasses import dataclass
from typing import List,Any,Union
from pynum.src import cu_manager


@dataclass(frozen=True)
class Device:
  SUPPORTED_DEVICES = {"CUDA","CPU"}
  device:str="CPU"
  def __repr__(self): return f"Device({self.device})"
  def __eq__(self, x): return self.device == x.device
  def __ne__(self, x): return self.device != x.device
  def to_str(self): return self.device


class CUDAManager:
  def __init__(self): self.__allocated_buffers = {}
  def allocate(self, data: List[Any], dtype):
    if not isinstance(data,list): raise TypeError("Data must be a list of integers")
    try:
      device_ptr = cu_manager.cuda_alloc(data)
      cu_manager.memcpy_htod(device_ptr,data)
      handle = id(device_ptr)
      self.__allocated_buffers[handle] = {
        "ptr": device_ptr,
        "size": len(data)
      }
      return handle
    except RuntimeError as e: raise RuntimeError(f"CUDA operation failed: {str(e)}")
  def free(self, handle):
    if handle not in self.__allocated_buffers: raise ValueError("Invalid handle")
    buffer_info = self.__allocated_buffers[handle]
    cu_manager.cuda_free(buffer_info["ptr"])
    del self.__allocated_buffers[handle]
  def ptr(self, handle): return self.__allocated_buffers[handle]["ptr"]
  def __del__(self):
    for handle in list(self.__allocated_buffers.keys()): self.free(handle)


