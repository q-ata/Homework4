import ctypes

import numpy as np

from ..util import qual_str
from .buffer import Buffer
from .c import CBufferFType, c_type, serialize_to_c
from .struct import TupleFType


class NumpyBuffer(Buffer):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the Buffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr = arr

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a NumpyBufferFType.
        """
        return NumpyBufferFType(self.arr.dtype.type, self.arr.shape)

    def length(self):
        return self.arr.size

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return self.arr.ndim

    @property
    def shape(self):
        """Shape of the tensor as a tuple."""
        return self.arr.shape

    @property
    def element_type(self):
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> "TupleFType":
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        return self.ftype.shape_type

    def load(self, indices: tuple[int, ...]):
        if len(indices) == self.ndim:
            return self.arr[*indices]
        raise ValueError(
            f"Expected {self.ndim} indices for {self.ndim}D array, got {len(indices)}"
        )

    def store(self, indices: tuple[int, ...], value):
        if len(indices) == self.ndim:
            self.arr[*indices] = value
        else:
            raise ValueError(
                f"Expected {self.ndim} indices for {self.ndim}D array, "
                f"got {len(indices)}"
            )

    def __str__(self):
        arr_str = str(self.arr).replace("\n", "")
        return f"np_buf({arr_str})"


numpy_buffer_allocators = {}
numpy_buffer_handles = []
numpy_buffer_types = {}


class NumpyBufferFType(CBufferFType):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class and TensorFType class.
    """

    def __init__(self, dtype: type, shape: tuple[int, ...]):
        self._dtype = np.dtype(dtype).type
        self._ndim = len(shape)
        self._shape = shape

    def __eq__(self, other):
        if not isinstance(other, NumpyBufferFType):
            return False
        return self._dtype == other._dtype and self._shape == other._shape

    def __str__(self):
        return f"np_buf_t({qual_str(self._dtype)}, shape={self._shape})"

    @property
    def shape(self) -> int:
        """Shape of the tensor."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return self._ndim

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return np.intp

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._dtype

    @property
    def shape_type(self) -> "TupleFType":
        """Shape type of the tensor. For NumpyBuffer, this is int for each dimension."""
        element_types = [int for _ in range(self._ndim)]
        return TupleFType("tuple", element_types)

    def __hash__(self):
        return hash((self._dtype, self._ndim))

    def __call__(self, shape, dtype: type | None = None):
        if dtype is None:
            dtype = self._dtype
        return NumpyBuffer(np.zeros(shape, dtype=dtype))

    def _c_type(self):
        if self not in numpy_buffer_types:
            # Get the ctypes pointer type for the element type
            data_t = ctypes.POINTER(c_type(self._dtype))
            dtype_name = np.dtype(self._dtype).name  # e.g. "int64", "float32"
            struct_name = f"CNumpyBuffer_{dtype_name}_{self._ndim}"
            shape_c_type = c_type(self.shape_type)

            CNumpyBufferType = type(
                struct_name,
                (ctypes.Structure,),
                {
                    "_fields_": [
                        ("arr", ctypes.py_object),
                        ("data", data_t),
                        ("length", ctypes.c_size_t),
                        ("shape", shape_c_type),
                    ]
                },
            )

            numpy_buffer_types[self] = CNumpyBufferType
        return numpy_buffer_types[self]

    def c_type(self):
        return self._c_type()

    def c_length(self, ctx, buf):
        return f"{ctx(buf)}.length"

    def c_data(self, ctx, buf):
        return f"{ctx(buf)}.data"

    def c_shape(self, ctx, buf):
        return f"{ctx(buf)}.shape"

    def _load(self, ctx, buf, indices):
        # For multi-dimensional arrays, calculate linear index from tuple of indices
        # linear_index = i0 * stride0 + i1 * stride1 + ... + in
        buf_code = ctx(buf)

        # Generate code to calculate linear index
        linear_idx_var = ctx.freshen("linear_idx")
        ctx.exec(f"{ctx.feed}size_t {linear_idx_var} = 0;")

        for dim in range(self.ndim):
            stride = 1
            for d in range(dim + 1, self.ndim):
                stride_code = f"{buf_code}.shape.element_{d}"
                stride = stride_code if stride == 1 else f"({stride}) * ({stride_code})"

            if stride == 1:
                ctx.exec(f"{ctx.feed}{linear_idx_var} += {ctx(indices[dim])};")
            else:
                ctx.exec(
                    f"{ctx.feed}{linear_idx_var} += ({ctx(indices[dim])}) * ({stride});"
                )

        return buf_code, linear_idx_var

    def c_vecgather(self, ctx, buf, base_offset_idxs, stride):
        buf_code = ctx(buf)

        # Generate code to calculate linear index
        linear_idx_var = ctx.freshen("linear_idx")
        ctx.exec(f"{ctx.feed}size_t {linear_idx_var} = 0;")

        for dim in range(self.ndim):
            offset_stride = 1
            for d in range(dim + 1, self.ndim):
                offset_stride_code = f"{buf_code}.shape.element_{d}"
                offset_stride = (
                    offset_stride_code
                    if offset_stride == 1
                    else f"({offset_stride}) * ({offset_stride_code})"
                )

            if offset_stride == 1:
                ctx.exec(f"{ctx.feed}{linear_idx_var} += {ctx(base_offset_idxs[dim])};")
            else:
                ctx.exec(
                    f"{ctx.feed}{linear_idx_var} += ({ctx(base_offset_idxs[dim])}) * ({offset_stride});"
                )

        gather_idx_str = ", ".join(
            [f"({i}*{ctx(stride)})" for i in reversed(range(8))]
        )
        return f"_mm512_i64gather_pd(_mm512_set_epi64({gather_idx_str}), {buf_code}.data + {linear_idx_var}, 8)"

    def c_load(self, ctx, buf, indices):
        buf_code, linear_idx_var = self._load(ctx, buf, indices)
        return f"{buf_code}.data[{linear_idx_var}]"

    def _store(self, ctx, buf, indices):
        # For multi-dimensional arrays, calculate linear index from tuple of indices
        buf_code = ctx(buf)

        # Generate code to calculate linear index
        linear_idx_var = ctx.freshen("linear_idx")
        ctx.exec(f"{ctx.feed}size_t {linear_idx_var} = 0;")

        for dim in range(self.ndim):
            stride = 1
            for d in range(dim + 1, self.ndim):
                stride_code = f"{buf_code}.shape.element_{d}"
                stride = stride_code if stride == 1 else f"({stride}) * ({stride_code})"

            if stride == 1:
                ctx.exec(f"{ctx.feed}{linear_idx_var} += {ctx(indices[dim])};")
            else:
                ctx.exec(
                    f"{ctx.feed}{linear_idx_var} += ({ctx(indices[dim])}) * ({stride});"
                )

        return buf_code, linear_idx_var

    def c_store(self, ctx, buf, indices, value):
        buf_code, linear_idx_var = self._store(ctx, buf, indices)
        ctx.exec(f"{ctx.feed}{buf_code}.data[{linear_idx_var}] = {ctx(value)};")

    def c_vecscatter(self, ctx, buf, base_offset_idxs, stride, value):
        buf_code = ctx(buf)

        # Generate code to calculate linear index
        linear_idx_var = ctx.freshen("linear_idx")
        ctx.exec(f"{ctx.feed}size_t {linear_idx_var} = 0;")

        for dim in range(self.ndim):
            offset_stride = 1
            for d in range(dim + 1, self.ndim):
                offset_stride_code = f"{buf_code}.shape.element_{d}"
                offset_stride = (
                    offset_stride_code
                    if offset_stride == 1
                    else f"({offset_stride}) * ({offset_stride_code})"
                )

            if offset_stride == 1:
                ctx.exec(f"{ctx.feed}{linear_idx_var} += {ctx(base_offset_idxs[dim])};")
            else:
                ctx.exec(
                    f"{ctx.feed}{linear_idx_var} += ({ctx(base_offset_idxs[dim])}) * ({offset_stride});"
                )

        scatter_idx_str = ", ".join(
            [f"({i}*{ctx(stride)})" for i in reversed(range(8))]
        )
        ctx.exec(
            f"{ctx.feed}_mm512_i64scatter_pd({buf_code}.data + {linear_idx_var}, _mm512_set_epi64({scatter_idx_str}), {ctx(value)}, 8);"
        )

    def c_alloc(self, ctx, shape):
        shape_code = [ctx(ctx.cache(f"dim_{i}", dim)) for i, dim in enumerate(shape)]

        if self not in numpy_buffer_allocators:

            @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t))
            def alloc_callback(shape):
                """
                A Python callback function that allocates a new NumPy array.
                """
                shape_tuple = tuple(shape[i] for i in range(self.ndim))
                buf = self(shape_tuple)
                numpy_buffer_handles.append(buf)  # prevent GC
                obj = self.serialize_to_c(buf)
                numpy_buffer_handles.append(obj)  # prevent GC
                return ctypes.addressof(obj)

            numpy_buffer_allocators[self] = alloc_callback

        alloc_name = f"alloc_{self._dtype.__name__}_{self.ndim}"

        ctx.add_global(alloc_name, numpy_buffer_allocators[self])

        shape_literal = f"(size_t[{self.ndim}]){{{', '.join(shape_code)}}}"

        t = ctx.ctype_name(self.c_type())

        return f"*({t}*){alloc_name}({shape_literal})"

    def serialize_to_c(self, obj):
        """
        Serialize the NumPy buffer to a C-compatible structure.
        """
        data_t = ctypes.POINTER(c_type(self._dtype))
        data = ctypes.cast(obj.arr.ctypes.data, data_t)
        length = obj.arr.size
        obj._self_obj = ctypes.py_object(obj)
        obj._c_shape = serialize_to_c(obj.shape_type, obj.shape)
        obj._c_buffer = self._c_type()(obj._self_obj, data, length, obj._c_shape)
        return obj._c_buffer

    def deserialize_from_c(self, obj, c_buffer):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """

    def construct_from_c(self, c_buffer):
        """
        Construct a NumpyBuffer from a C-compatible structure.
        """
        return c_buffer.arr
