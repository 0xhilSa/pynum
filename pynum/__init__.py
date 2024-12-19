import subprocess
from .vector import *
from . import dtypes


def is_cuda_available():
  try:
    result = subprocess.run(
      ["nvidia-smi"],
      stdout = subprocess.PIPE,
      stderr = subprocess.PIPE,
      text = True
    )
    if result.returncode == 0: return True
    else: return False
  except FileExistsError: return False
