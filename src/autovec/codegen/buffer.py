from abc import ABC, abstractmethod
from typing import Any

from .. import algebra
from ..algebra import query_property
from ..symbolic import FType, FTyped
from .struct import TupleFType


class Buffer(FTyped, ABC):
    """
    Abstract base class for buffer-like data structures. Buffers support random access
    for reading and writing of elements.
    """

    @abstractmethod
    def __init__(self, length: int, dtype: type): ...

    @abstractmethod
    def length(self):
        """
        Return the length of the buffer.
        """
        ...

    @property
    def element_type(self):
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self.ftype.element_type

    @property
    def length_type(self):
        """
        Return the type of indices used to access elements in the buffer.
        This is typically an integer type.
        """
        return self.ftype.length_type()

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return self.ftype.ndim

    @property
    @abstractmethod
    def shape(self):
        """Shape of the tensor as a tuple."""
        ...

    @property
    def shape_type(self) -> "TupleFType":
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        return self.ftype.shape_type

    @abstractmethod
    def load(self, indices: tuple[int, ...]): ...

    @abstractmethod
    def store(self, indices: tuple[int, ...], val): ...


def length_type(arg: Any):
    """The length type of the given argument. The length type is the type of
    the value returned by len(arg).

    Args:
        arg: The object to determine the length type for.

    Returns:
        The length type of the given object.

    Raises:
        AttributeError: If the length type is not implemented for the given type.
    """
    if hasattr(arg, "length_type"):
        return arg.length_type
    return query_property(arg, "length_type", "__attr__")


def element_type(arg: Any):
    return algebra.element_type(arg)


class BufferFType(FType):
    """
    Abstract base class for the ftype of arguments. The ftype defines how the
    data structures store data, and can construct a data structure with the call method.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Create an instance of an object in this ftype with the given arguments.
        """
        ...

    @property
    @abstractmethod
    def element_type(self):
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        ...

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return int

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        ...

    @property
    @abstractmethod
    def shape_type(self) -> "TupleFType":
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        ...
