import os, sys, platform, subprocess
from dataclasses import dataclass


def cpu_info():
  info = {}
  if os.name == "posix":
    try:
      if sys.platform == "linux":
        with open("/proc/cpuinfo","r") as f:
          for line in f:
            if line.strip():
              key, _, value = line.partition(":")
              info[key.strip()] = value.strip()
      elif sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-a"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
          if "machdep.cpu" in line:
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
    except Exception as e: info["error"] = str(e)
  elif os.name == "nt":
    try:
      import wmi
      c = wmi.WMI()
      for processor in c.Win32_Processor():
        info["Name"] = processor.Name
        info["NumberOfCores"] = processor.NumberOfCores
        info["MaxClockSpeed"] = processor.MaxClockSpeed
    except Exception as e: info["error"] = str(e)
  return info
def cuda_info():
  info = {}
  try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
      for line in result.stdout.split("\n"):
        if "release" in line: info["version"] = line.strip()
    else: info["error"] = "CUDA not installed or nvcc not in PATH."
  except FileNotFoundError: info["error"] = "nvcc command not found. Ensure CUDA is installed and added to PATH."
  except Exception as e: info["error"] = str(e)
  return info
