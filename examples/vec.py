from pynum.vector import Vector

x = Vector([1+3j,2-5j,-0.7+12j,-0.9-1j], dtype=complex, device="CUDA")

print(x)
print(x.numpy())
print(x.device)

x.cpu()   # vector from CUDA to CPU

print(x)
print(x.numpy())
print(x.device)

print(x[0])
print(x[0].numpy())
