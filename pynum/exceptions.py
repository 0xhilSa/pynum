class VectorError(Exception): pass
class DTypeError(VectorError): pass
class DeviceError(VectorError): pass
class ImmutableVectorError(VectorError): pass
class VectorIndexError(VectorError): pass
