#!/bin/bash

CC=gcc
NVCC=nvcc
PYTHON_VERSION=3.10
PYTHON_INCLUDE=$(python${PYTHON_VERSION}-config --includes)
PYTHON_LIBS=$(python${PYTHON_VERSION}-config --ldflags)
CUDA_PATH=/usr/local/cuda

# source files
HOST_SRC="./pynum/csrc/host.c"
CUDA_SRC="./pynum/csrc/pycu.cu"
HOST_OUT="./pynum/csrc/host.so"
CUDA_OUT="./pynum/csrc/pycu.so"

# spinner function
spinner() {
    local pid=$1
    local delay=0.1
    local spin='|/-\'
    local i=0

    while kill -0 $pid 2>/dev/null; do
        printf "\rCompiling C and CUDA source file %s" "${spin:i++%4:1}"
        sleep $delay
    done
}

# compile host.c
$CC -Wall -shared -fPIC $HOST_SRC -o $HOST_OUT $PYTHON_INCLUDE $PYTHON_LIBS &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

# compile pycu.cu
$NVCC -shared -Xcompiler -fPIC -I$CUDA_PATH/include $PYTHON_INCLUDE \
    -o $CUDA_OUT -L$CUDA_PATH/lib64 -lcudart $CUDA_SRC &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

# clear spinner and show success message
echo -ne "\r\033[KCompiled Successfully!\n"

# check if pynum is already installed
if python3 -c "import pynum" 2>/dev/null; then
  read -p "pynum is already installed. Do you want to reinstall it? (y/N): " choice
  case "$choice" in
    y|Y)
      echo "Reinstalling pynum..."
      pip uninstall -y pynum
      ;;
    *)
      echo "Aborting installation."
      exit 1
      ;;
  esac
fi

# install pynum from source
python3 -m build
cd dist/ && pip install *.whl && cd ..
rm -rf dist/ build/ pynum.egg-info/
