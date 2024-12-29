from enum import auto, Enum
import numpy as np


class Ops(Enum):
  # uniary ops
  LOG = auto(); LOG10 = auto(); LOG2 = auto()
  SIN = auto(); COS = auto(); TAN = auto()
  SINH = auto(); COSH = auto(); TANH = auto()
  RECI = auto(); NEG = auto(); SQRT = auto()
  CBRT = auto(); EXP = auto(); EXP2 = auto()
  NOT = auto()

  # binary ops
  ADD = auto(); SUB = auto(); MUL = auto(); TDIV = auto(); FDIV = auto(); MOD = auto(); POW = auto()
  MAX = auto(); MIN = auto()
  AND = auto(); OR = auto(); XOR = auto()

class GroupOp:
  Unary = {Ops.LOG, Ops.LOG10, Ops.LOG2, Ops.SIN, Ops.COS, Ops.TAN, Ops.SINH, Ops.COSH, Ops.TANH, Ops.RECI, Ops.NEG, Ops.SQRT, Ops.CBRT, Ops.EXP, Ops.EXP2}
  BINARY = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.TDIV, Ops.FDIV, Ops.MOD, Ops.POW, Ops.MAX, Ops.MIN, Ops.AND, Ops.OR, Ops.XOR}

ops_mapping = {
  # binary ops
  Ops.LOG: np.log,
  Ops.LOG10: np.log10,
  Ops.LOG2: np.log2,
  Ops.SIN: np.sin,
  Ops.COS: np.cos,
  Ops.TAN: np.tan,
  Ops.SINH: np.sinh,
  Ops.COSH: np.cosh,
  Ops.TANH: np.tanh,
  Ops.EXP: np.exp,
  Ops.EXP2: np.exp2,
  Ops.SQRT: lambda x : x ** 0.5,
  Ops.CBRT: lambda x : x ** (1/3),
  Ops.RECI: lambda x : 1/x,
  Ops.NEG: lambda x : -x,
  Ops.NOT: lambda x : ~x,

  # binary ops
  Ops.ADD: lambda x,y : x + y,
  Ops.SUB: lambda x,y : x - y,
  Ops.MUL: lambda x,y : x * y,
  Ops.TDIV: lambda x,y : x / y,
  Ops.FDIV: lambda x,y : x // y,
  Ops.MOD: lambda x,y : x % y,
  Ops.POW: lambda x,y : x ** y,
  Ops.MAX: lambda x,y : max(x,y),
  Ops.MIN: lambda x,y : min(x,y),
  Ops.AND: lambda x,y : x & y,
  Ops.OR: lambda x,y : x | y,
  Ops.XOR: lambda x,y : x ^ y
}

