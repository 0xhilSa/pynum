class DType:
  _allowed_formats = {
    "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d", "F", "D", "g", "G", "?"
  }
  def __init__(self, name:str, fmt:str):
    if not isinstance(fmt, str) or not fmt: raise ValueError("fmt must be a non-empty string")
    if not isinstance(name, str) or not name: raise ValueError("name must be a non-empty string")
    if fmt not in self._allowed_formats: raise ValueError(f"Invalid fmt '{fmt}'. Allowed formats: {self._allowed_formats}")
    self.__fmt = fmt
    self.__name = name
  def __repr__(self): return f"<DType(name='{self.__name}', fmt='{self.__fmt}')>"
  def __del__(self): del self
  @property
  def name(self): return self.__name
  @property
  def fmt(self): return self.__fmt


# pre-defined dtypes for backend
char = int8 = DType("char", "b")
uchar = uint8 = DType("unsigned char", "B")
short = int16 = DType("short", "h")
ushort = uint16 = DType("unsigned short", "H")
int_ = int32 = DType("int", "i")
uint_ = uint32 = DType("unsigned int", "I")
long = int64 = DType("long", "l")
ulong = uint64 = DType("unsigned long", "L")
longlong = DType("long long", "q")
ulonglong = DType("unsigned long long", "Q")
float_ = float32 = DType("float", "f")
double = float64 = DType("double", "d")
longdouble = DType("long double", "g")
floatcomplex = complex64 = DType("float complex", "F")
doublecomplex = complex128 = DType("double complex", "D")
longdoublecomplex = complex256 = DType("long double complex", "G")
bool_ = DType("bool", "?")
