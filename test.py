from pynum.src.cu_manager import cuda_alloc, cuda_free, cuda_query_free_mem

x = [1,2,3,4,5]
ptr = cuda_alloc(x)
#print(id(ptr))
cuda_free(ptr)

