class VectorError(Exception): pass
class InvalidDTypeError(VectorError): pass
class DeviceError(VectorError): pass
class ImmutableVectorError(VectorError): pass
class VectorIndexError(VectorError): pass
