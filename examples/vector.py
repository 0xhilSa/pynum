from pynum.vector import Vector
from pynum import dtype

x = Vector([1,2,3,4,5,6,7], dtype=dtype.int8, device="cpu")       # vector on CPU and mutable
y = Vector([1,2,3,4,5,6,7], dtype=dtype.int64, device="cuda")     # vector on GPU and mutable

a = Vector([2.3, 3.14, 2.71, 5.61, 9.8, 1.33], dtype=dtype.float32, device="cpu", const=True)     # vector on CPU and immutable
b = Vector([2.3, 3.14, 2.71, 5.61, 9.8, 1.33], dtype=dtype.float64, device="cuda", const=True)    # vector on GPU and immutable
c = Vector([2.3, 3.14, 2.71, 5.61, 9.8, 1.33], dtype=dtype.float128, device="cuda", const=True)   # vector on GPU and immutable

print(x, x.numpy())
print(y, y.numpy())
print(a, a.numpy())
print(b, b.numpy())
print(c, c.numpy())
