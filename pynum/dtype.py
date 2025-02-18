from dataclasses import dataclass
from typing import Final, Literal


Fmts = Literal["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d", "g", "F", "D", "G"]
__fmts = ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d", "g", "F", "D", "G"]
__dtype_names = [
  "boolean",
  "char",
  "unsigned char",
  "short",
  "unsigned short",
  "int",
  "unsigned int",
  "long",
  "unsigned long",
  "long long",
  "unsigned long long",
  "float",
  "double",
  "long double",
  "float complex",
  "double complex",
  "long double complex"
]
__dtypes_dict = {name: i for i, name in enumerate(__dtype_names)}

@dataclass(frozen=True, eq=False)
class DType:
  name:str
  fmt:Fmts
  priority:int
  @staticmethod
  def new(name:str, fmt:Fmts, priority:int): return DType(name, fmt, priority)
  def __repr__(self): return f"<DType(name='{self.name}', fmt='{self.fmt}', priority={self.priority})>"
  def name(self): return self.name
  def fmt(self): return self.fmt
  def priority(self): return self.priority

def get_by_fmt(fmt: str) -> DType:
  try:
    index = __fmts.index(fmt)
    name = __dtype_names[index]
    return DType(name, fmt, index)
  except ValueError: raise ValueError(f"Invalid format specifier: {fmt}")

def get_by_name(name: str) -> DType:
  try:
    index = __dtypes_dict[name]
    return DType(name, __fmts[index], index)
  except KeyError: raise ValueError(f"Invalid dtype name: {name}")

def get_by_priority(priority: int) -> DType:
  if 0 <= priority < len(__dtype_names): return DType(__dtype_names[priority], __fmts[priority], priority)
  raise ValueError(f"Invalid priority: {priority}")

# pre-defined dtypes for backend
boolean:Final[DType] = DType.new("boolean", "?", 0)
int8:Final[DType] = DType.new("char", "b", 1)
uint8:Final[DType] = DType.new("unsigned char", "B", 2)
int16:Final[DType] = DType.new("short", "h", 3)
uint16:Final[DType] = DType.new("unsigend short", "H", 4)
int32:Final[DType] = DType.new("int", "i", 5)
uint32:Final[DType] = DType.new("unsigned int", "I", 6)
int64:Final[DType] = DType.new("long", "l", 7)
uint64:Final[DType] = DType.new("unsigned long", "L", 8)
longlong:Final[DType] = DType.new("long long", "q", 9)
ulonglong:Final[DType] = DType.new("unsigned long long", "Q", 10)
float32:Final[DType] = DType.new("float", "f", 11)
float64:Final[DType] = DType.new("double", "d", 12)
longdouble:Final[DType] = DType.new("long double", "g", 13)
complex64:Final[DType] = DType.new("float complex", "F", 14)
complex128:Final[DType] = DType.new("double complex", "D", 15)
complex256:Final[DType] = DType.new("long double complex", "G", 16)

SIGNED_INTEGERS = (int, int8, int16, int32, int64, longlong)
UNSIGNED_INTEGERS = (uint8, uint16, uint32, uint64, ulonglong)
FLOATING = (float, float32, float64, longdouble)
COMPLEX = (complex, complex64, complex128, complex256)
BOOL = (bool, boolean)
VALID_DTYPES = SIGNED_INTEGERS + UNSIGNED_INTEGERS + FLOATING + COMPLEX + BOOL
