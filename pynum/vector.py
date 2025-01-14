from __future__ import annotations
from typing import List, Any, Type, Tuple, Optional, Union
import numpy as np
from .cuda import *
from .ops import Ops, GroupOp, ops_mapping


DEVICE_SUPPORTED = ["CPU", "CUDA"]
FLOATING = [float, np.float16, np.float32, np.float64, np.float128]
INTEGER = [int, np.int8, np.int16, np.int32, np.int64]
COMPLEX = [complex, np.complex128]
ALL = FLOATING + INTEGER + COMPLEX


class Vector:
  def __init__(self, array:List): pass
  def __repr__(self): pass