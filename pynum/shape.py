from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Shape:
  dims:Tuple[int,...]
  def __repr__(self): return f"Shape{self.dims}"
  def __len__(self): return self.dims[0] if self.dims else 0
  def __eq__(self, x): return self.dims == x.dims
  def __ne__(self, x): return not self.__eq__(x)
