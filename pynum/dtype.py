# `dtype.py` will provide the custom dtypes for
# the arithmetic ops on vectors in back end C/C++

from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Literal

Fmts = Literal["?","b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d", "g", "F", "D", "G"]

@dataclass(frozen=True, eq=True)
class DType:
  name:str
  fmt:Fmts
  priority:int
  signed:bool
  @classmethod
  def new(cls, name:str, fmt:Fmts, priority:int, signed:bool): return DType(name, fmt, priority, signed)
  def __repr__(self): return f"<DType(name='{self.name}' fmt='{self.fmt}' priority={self.priority} signed={self.signed})>"
  def get_name(self): return self.name
  def get_fmt(self): return self.fmt
  def get_priority(self): return self.priority
  def get_signed(self): return self.signed


boolean:Final[DType] = DType.new("bool", "?", 0, False)
int8:Final[DType] = DType.new("char", "b", 1, True)
uint8:Final[DType] = DType.new("unsigned char", "B", 2, False)
int16:Final[DType] = DType.new("short", "h", 3, True)
uint16:Final[DType] = DType.new("unsigned short", "H", 4, False)
int32:Final[DType] = DType.new("int", "i", 5, True)
uint32:Final[DType] = DType.new("unsigned int", "I", 6, False)
int64:Final[DType] = DType.new("long", "l", 7, True)
uint64:Final[DType] = DType.new("unsigned long", "L", 8, False)
longlong:Final[DType] = DType.new("long long", "q", 9, True)
ulonglong:Final[DType] = DType.new("unsigned long long", "Q", 10, False)
float32:Final[DType] = DType.new("float", "f", 11, True)
float64:Final[DType] = DType.new("double", "d", 12, True)
float128:Final[DType] = DType.new("long double", "g", 13, True)
complex64:Final[DType] = DType.new("float complex", "F", 14, True)
complex128:Final[DType] = DType.new("double complex", "D", 15, True)
complex256:Final[DType] = DType.new("long double complex", "G", 16, True)

SIGNED = (int8, int16, int32, int64, longlong, float32, float64, float128, complex64, complex128, complex256)
UNSIGNED = (uint8, uint16, uint32, uint64, ulonglong)
INTEGER = (int8, uint8, int16, uint16, int32, uint32, int64, uint64, longlong, ulonglong)
FLOATING = (float32, float64, float128)
COMPLEX = (complex64, complex128, complex256)
BOOLEAN = (boolean,)
BUILTIN = (int, float, complex, bool)
VALID_DTYPES = INTEGER + FLOATING + COMPLEX + BOOLEAN + BUILTIN

def by_name(name:str):
  for dtype in VALID_DTYPES:
    if isinstance(dtype,DType) and dtype.name == name: return dtype
  return None

def by_fmt(fmt:Fmts):
  for dtype in VALID_DTYPES:
    if isinstance(dtype,DType) and dtype.fmt == fmt: return dtype
  return None

def by_priority(priority:int):
  for dtype in VALID_DTYPES:
    if isinstance(dtype,DType) and dtype.priority == priority: return dtype
  return None
