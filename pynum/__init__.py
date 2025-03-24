"""

PyNum (v0.0.1)
==============

provides:
  - GPU acceleration using CUDA
  - Simplicity

modules:
  - csrc.host
  - csrc.pycu
  - dtype
  - vector


>>> from pynum.vector import Vector
>>> from pynum import dtype
>>> x = Vector([1,2,3,4,5,6,7,8,9,10], dtype=dtype.longlong, device="cpu", const=False)   # mutable vector on a host(cpu) device
>>> x
<Vector(length=10, dtype='long long', device='cpu', const=False)>
>>> x.numpy()
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
>>> y = Vector([1,2,3,4,5,6,7,8,9,10], dtype=dtype.complex128, device="cuda", const=True)    # immuatable vector on a gpu(cuda) device
<Vector(length=10, dtype='double complex', device='cuda', const=False)>
>>> y.numpy()
array([ 1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  5.+0.j,  6.+0.j,  7.+0.j,
        8.+0.j,  9.+0.j, 10.+0.j])


NOTE: PyNum is under construction, so bugs and errors are normal :)
"""
