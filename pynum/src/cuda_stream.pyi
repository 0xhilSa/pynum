from typing import List, Any, Type

def alloc_short(x:List) -> int: ...
def alloc_int(x:List) -> int: ...
def alloc_long(x:List) -> int: ...
def alloc_float(x:List) -> int: ...
def alloc_double(x:List) -> int: ...
def alloc_complex(x:List) -> int: ...
def alloc_bool(x:List) -> int: ...

def memcpy_htod_short(ptr:int, x:List) -> None: ...
def memcpy_htod_int(ptr:int, x:List) -> None: ...
def memcpy_htod_long(ptr:int, x:List) -> None: ...
def memcpy_htod_float(ptr:int, x:List) -> None: ...
def memcpy_htod_double(ptr:int, x:List) -> None: ...
def memcpy_htod_complex(ptr:int, x:List) -> None: ...
def memcpy_htod_bool(ptr:int, x:List) -> None: ...

def memcpy_dtoh_short(ptr:int, length:int) -> List: ...
def memcpy_dtoh_int(ptr:int, length:int) -> List: ...
def memcpy_dtoh_long(ptr:int, length:int) -> List: ...
def memcpy_dtoh_float(ptr:int, length:int) -> List: ...
def memcpy_dtoh_double(ptr:int, length:int) -> List: ...
def memcpy_dtoh_complex(ptr:int, length:int) -> List: ...
def memcpy_dtoh_bool(ptr:int, length:int) -> List: ...

def get_value_short(ptr:int, index:int) -> Type: ...
def get_value_int(ptr:int, index:int) -> Type: ...
def get_value_long(ptr:int, index:int) -> Type: ...
def get_value_float(ptr:int, index:int) -> Type: ...
def get_value_double(ptr:int, index:int) -> Type: ...
def get_value_complex(ptr:int, index:int) -> Type: ...
def get_value_bool(ptr:int, index:int) -> Type: ...

def get_slice_short(ptr:int, start:int, stop:int, step:int) -> List: ...
def get_slice_int(ptr:int, start:int, stop:int, step:int) -> List: ...
def get_slice_long(ptr:int, start:int, stop:int, step:int) -> List: ...
def get_slice_float(ptr:int, start:int, stop:int, step:int) -> List: ...
def get_slice_double(ptr:int, start:int, stop:int, step:int) -> List: ...
def get_slice_complex(ptr:int, start:int, stop:int, step:int) -> List: ...
def get_slice_bool(ptr:int, start:int, stop:int, step:int) -> List: ...

def set_value_short(ptr:int, index:int, value:Any) -> None: ...
def set_value_int(ptr:int, index:int, value:Any) -> None: ...
def set_value_long(ptr:int, index:int, value:Any) -> None: ...
def set_value_float(ptr:int, index:int, value:Any) -> None: ...
def set_value_double(ptr:int, index:int, value:Any) -> None: ...
def set_value_complex(ptr:int, index:int, value:Any) -> None: ...
def set_value_bool(ptr:int, index:int, value:Any) -> None: ...

def add_short(ptr1:int, ptr2:int, length:int) -> int: ...
def add_int(ptr1:int, ptr2:int, length:int) -> int: ...
def add_long(ptr1:int, ptr2:int, length:int) -> int: ...
def add_float(ptr1:int, ptr2:int, length:int) -> int: ...
def add_double(ptr1:int, ptr2:int, length:int) -> int: ...
def add_complex(ptr1:int, ptr2:int, length:int) -> int: ...

def free(ptr:int) -> None: ...
def count_device() -> int: ...
def select_device() -> None: ...
