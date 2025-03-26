# all the functions are defined at `./host.c`

from typing import List, Any

def array(array1D:List[Any], fmt:str) -> List[Any]: ...
def toList(pointer:Any, length:int, fmt:str) -> List: ...
def getitem_index(pointer:Any, length:int, index:int, fmt:str) -> Any: ...
def getitem_slice(pointer:Any, length:int, start:int, stop:int, step:int, fmt:str) -> Any: ...
def setitem_index(pointer:Any, value:Any, length:int, index:int, fmt:str) -> None: ...
def setitem_slice(pointer:Any, values:List[Any], length:int, start:int, stop:int, step:int, fmt:str) -> None: ...
