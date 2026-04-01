from . import nodes as asm
from ..simple_lang import nodes as smpl
from ..symbolic.environment import Namespace
import operator


class LowererError(Exception):
    pass


class SimpleLangToAssembly:
    """Converts SimpleLang IR nodes to Assembly IR nodes."""

    def __init__(self, vector_width):
        self.vector_width = vector_width
        self.namespace = Namespace()
        self.var_map = {}
        self.vec_idx_name = "vec_i"

    def _get_vector_idx(
        self, indices: tuple[smpl.SimpleLangExpression | smpl.VectorIndex, ...]
    ) -> smpl.VectorIndex | None:
        vector_idx = []
        for idx in indices:
            if isinstance(idx, smpl.VectorIndex):
                vector_idx.append(idx)

        if len(vector_idx) == 1:
            return vector_idx[0]
        elif len(vector_idx) > 1:
            raise LowererError("Encountered multiple vector indexes")
        else:
            None

    def _collect_vector_indexes(
        self, node: smpl.SimpleLangNode
    ) -> set[smpl.VectorIndex]:
        vec_idx_set = set()
        match node:
            case smpl.Store(_, indices, val):
                vec_idx_set.add(self._get_vector_idx(indices))
                vec_idx_set.update(self._collect_vector_indexes(val))
            case smpl.Load(_, indices):
                val = self._get_vector_idx(indices)
                if val is not None:
                    vec_idx_set.add(val)
            case smpl.Call(_, args):
                for arg in args:
                    vec_idx_set.update(self._collect_vector_indexes(arg))

        return vec_idx_set

    def _resolve_scalar_inst(
        self,
        node: smpl.SimpleLangNode,
        vector_idx: asm.Index,
        vec_idx_to_loop_idx_dict: dict[smpl.VectorIndex, asm.AssemblyNode],
    ):
        match node:
            case smpl.Literal(_):
                return self(node)
            case (smpl.Call() | smpl.Load()) as inst if not inst.is_vectorized():
                return self(node)

            case smpl.Call(op, args) as inst if inst.is_vectorized():
                return asm.Call(
                    asm.Literal(op.val),
                    tuple(
                        self._resolve_scalar_inst(
                            arg, vector_idx, vec_idx_to_loop_idx_dict
                        )
                        for arg in args
                    ),
                )

            case smpl.Load(buf, indices) as inst if inst.is_vectorized():
                new_indices = []
                for idx in indices:
                    if isinstance(idx, smpl.VectorIndex):
                        new_indices.append(vec_idx_to_loop_idx_dict[idx])
                    else:
                        new_indices.append(self(idx))
                return asm.Load(self(buf), tuple(new_indices))
            case _:
                raise NotImplementedError(f"Unrecognized node type: {type(node)}")

    def _resolve_vector_inst(
        self,
        node: smpl.SimpleLangNode,
        vector_idx: asm.Index,
        vec_idx_to_loop_idx_dict: dict[smpl.VectorIndex, asm.AssemblyNode],
    ):
        match node:
            case (smpl.Call() | smpl.Load()) as inst if not inst.is_vectorized():
                return asm.VectorBroadcast(self(node))

            case smpl.Literal(val):
                return asm.VectorBroadcast(asm.Literal(val))

            case smpl.Call(op, args) as inst if inst.is_vectorized():
                return asm.VectorCall(
                    asm.Literal(op.val),
                    tuple(
                        self._resolve_vector_inst(
                            arg, vector_idx, vec_idx_to_loop_idx_dict
                        )
                        for arg in args
                    ),
                )

            case smpl.Load(buf, indices) as inst if inst.is_vectorized():
                new_indices = []
                vector_idx_dim = None
                for dim, idx in enumerate(indices):
                    if isinstance(idx, smpl.VectorIndex):
                        vector_idx_dim = dim
                        new_indices.append(vec_idx_to_loop_idx_dict[idx])
                    else:
                        new_indices.append(self(idx))

                # Conditionally codegen based on whether strided vector access
                load_vec_idx = self._get_vector_idx(indices)

                # Compute the gather stride
                gather_stride = load_vec_idx.stride
                for i in range(vector_idx_dim + 1, node.buffer.type.ndim):
                    gather_stride = gather_stride * node.buffer.type.shape[i]

                return asm.VectorGather(
                    self(buf), tuple(new_indices), asm.Literal(gather_stride)
                )
            case _:
                raise NotImplementedError(f"Unrecognized node type: {type(node)}")

    def _freshen_children(self, node: smpl.SimpleLangNode):
        match node:
            case smpl.Load(buf, _):
                self.namespace.freshen(buf.name)
            case smpl.Call(_, args):
                for arg in args:
                    self._freshen_children(arg)

    def __call__(self, node: smpl.SimpleLangNode) -> asm.AssemblyNode:
        match node:
            case smpl.Function(name, args, body):
                return asm.Module(
                    (
                        asm.Function(
                            self(name), tuple(self(arg) for arg in args), self(body)
                        ),
                    )
                )

            case smpl.Block(bodies):
                output = []
                for body in bodies:
                    res = self(body)
                    if isinstance(res, list):
                        output.extend(res)
                    else:
                        output.append(res)

                return asm.Block(tuple(output))

            case smpl.ForLoop(lvl, start, end, stride, body):
                return asm.ForLoop(
                    self(lvl),
                    self(start),
                    self(end),
                    self(stride),
                    self(body),
                )

            case smpl.Return(arg):
                return asm.Return(self(arg))

            case smpl.Literal(val):
                return asm.Literal(val)

            case smpl.Variable(name, type):
                # record the variable name in the namespace so that the
                # vector temporaries do not conflict.
                self.namespace.freshen(name)
                return asm.Variable(name, type)

            case smpl.Index(name):
                # record the index in the namespace so that the
                # vector loop indexes do not conflict.
                self.namespace.freshen(name)
                return asm.Index(name)

            case smpl.Store(buffer, indices, value) as inst if not inst.is_vectorized():
                return asm.Store(
                    self(buffer), tuple(self(idx) for idx in indices), self(value)
                )

            case smpl.Call(op, args) as inst if not inst.is_vectorized():
                return asm.Call(self(op), tuple(self(arg) for arg in args))

            case smpl.Load(buffer, indices) as inst if not inst.is_vectorized():
                return asm.Load(self(buffer), tuple(self(idx) for idx in indices))

            case smpl.Store(buffer, indices, value) as inst if inst.is_vectorized():
                assembly_out = []

                # Create temporary buffer, but before we do this
                # we look through the RHS and ensure we record all the
                # variable names to ensure that the temporary name does not collide
                self._freshen_children(value)
                self.namespace.freshen(buffer.name)
                temp_buf = asm.Variable(
                    self.namespace.freshen(buffer.name), buffer.type
                )

                assembly_out.append(asm.Allocate(temp_buf))

                # New Idx name
                new_idx = asm.Index(self.namespace.freshen(self.vec_idx_name))

                # Get a list of all vector indices in this store tree so that
                # we can decide the offsets when creating loops
                vec_idx_set = self._collect_vector_indexes(node)

                # Verify that all vectors are of the same size
                vec_range_list = []
                for vec_idx in vec_idx_set:
                    vec_range_list.append(
                        int((vec_idx.end - vec_idx.start) / vec_idx.stride)
                    )

                if not all(item == vec_range_list[0] for item in vec_range_list):
                    raise LowererError(
                        "All vector operations are not of the same size!"
                    )

                # Generate normalized representations of indexes
                vec_idx_to_loop_idx_dict = dict()
                for vec_idx in vec_idx_set:
                    vec_idx_to_loop_idx_dict[vec_idx] = asm.Call(
                        asm.Literal(operator.add),
                        (
                            asm.Literal(vec_idx.start),
                            asm.Call(
                                asm.Literal(operator.mul),
                                (new_idx, asm.Literal(vec_idx.stride)),
                            ),
                        ),
                    )

                # Replace vector index with a new index for the loop that we introduce
                new_indices = []
                vector_idx_dim = None
                for dim, idx in enumerate(indices):
                    if isinstance(idx, smpl.VectorIndex):
                        new_indices.append(vec_idx_to_loop_idx_dict[idx])
                        vector_idx_dim = dim
                    else:
                        new_indices.append(self(idx))
                new_indices = tuple(new_indices)

                # Compute loop bounds for store operation
                loop_start = 0
                loop_end = vec_range_list[0]
                vectorized_distance = (
                    (int)((loop_end - loop_start) / self.vector_width)
                ) * self.vector_width
                is_remainder_loop_needed = loop_start + vectorized_distance < loop_end

                # Conditionally codegen based on whether strided vector access
                store_vec_idx = self._get_vector_idx(indices)

                # Compute the scatter stride
                scatter_stride = store_vec_idx.stride
                for i in range(vector_idx_dim + 1, node.buffer.type.ndim):
                    scatter_stride = scatter_stride * node.buffer.type.shape[i]

                # perform actual operation with temporary
                assembly_out.append(
                    asm.ForLoop(
                        new_idx,
                        asm.Literal(loop_start),
                        asm.Literal(loop_start + vectorized_distance),
                        asm.Literal(self.vector_width),
                        asm.Block(
                            (
                                asm.VectorScatter(
                                    temp_buf,
                                    new_indices,
                                    asm.Literal(scatter_stride),
                                    self._resolve_vector_inst(
                                        value, new_idx, vec_idx_to_loop_idx_dict
                                    ),
                                ),
                            )
                        ),
                    )
                )

                # Reminder loops
                if is_remainder_loop_needed:
                    assembly_out.append(
                        asm.ForLoop(
                            new_idx,
                            asm.Literal(loop_start + vectorized_distance),
                            asm.Literal(loop_end),
                            asm.Literal(1),
                            asm.Block(
                                (
                                    asm.Store(
                                        temp_buf,
                                        new_indices,
                                        self._resolve_scalar_inst(
                                            value, new_idx, vec_idx_to_loop_idx_dict
                                        ),
                                    ),
                                )
                            ),
                        )
                    )

                assembly_out.append(
                    asm.ForLoop(
                        new_idx,
                        asm.Literal(loop_start),
                        asm.Literal(loop_start + vectorized_distance),
                        asm.Literal(self.vector_width),
                        asm.Block(
                            (
                                asm.VectorScatter(
                                    self(buffer),
                                    new_indices,
                                    asm.Literal(scatter_stride),
                                    asm.VectorGather(
                                        temp_buf,
                                        new_indices,
                                        asm.Literal(scatter_stride),
                                    ),
                                ),
                            )
                        ),
                    )
                )

                # add reminder loops
                if is_remainder_loop_needed:
                    assembly_out.append(
                        asm.ForLoop(
                            new_idx,
                            asm.Literal(loop_start + vectorized_distance),
                            asm.Literal(loop_end),
                            asm.Literal(1),
                            asm.Block(
                                (
                                    asm.Store(
                                        self(buffer),
                                        new_indices,
                                        asm.Load(temp_buf, new_indices),
                                    ),
                                )
                            ),
                        )
                    )

                return assembly_out

            case _:
                raise NotImplementedError(f"Unrecognized node type: {type(node)}")
