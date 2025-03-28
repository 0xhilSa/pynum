from pynum.vector import Vector
from pynum import dtype

x = Vector([1,2,3,4,5,6], dtype=dtype.longlong, device="cpu")
y = Vector([4,1,0,9,2,3], dtype=dtype.longlong, device="cpu")
z = x + y
print(x, x.numpy())
print(y, x.numpy())
print(z, z.numpy())
