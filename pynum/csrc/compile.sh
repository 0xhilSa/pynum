#!/bin/bash

CC=gcc
PYTHON_VERSION=3.10
PYTHON_INCLUDE=$(python${PYTHON_VERSION}-config --includes)
PYTHON_LIBS=$(python${PYTHON_VERSION}-config --ldflags)

SRC="host.c"
OUT="host.so"

$CC -Wall -shared -fPIC $SRC -o $OUT $PYTHON_INCLUDE $PYTHON_LIBS

