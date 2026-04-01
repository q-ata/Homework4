from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from autovec.algebra.tensor import shape_type

from ..algebra import return_type
from ..codegen.buffer import element_type, length_type
from ..symbolic import Context, Term, TermTree, literal_repr
from ..util import qual_str


class AssemblyNode(Term):
    """
    AssemblyNode

    Represents a Assembly IR node. Assembly is the final intermediate
    representation before code generation (translation to the output language).
    It is a low-level imperative description of the program, with control flow,
    linear memory regions called "buffers", and explicit memory management.
    """

    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head, *args):
        """Creates a term with the given head and arguments."""
        return head.from_children(*args)

    @classmethod
    def from_children(cls, *children):
        """
        Creates a term from the given children. This is used to create terms
        from the children of a node.
        """
        return cls(*children)

    def __str__(self):
        """Returns a string representation of the node."""
        ctx = AssemblyPrinterContext()
        ctx(self)
        return ctx.emit()


class AssemblyTree(AssemblyNode, TermTree):
    @property
    def children(self):
        """Returns the children of the node."""
        raise Exception(f"`children` isn't supported for {self.__class__}.")


class AssemblyExpression(AssemblyNode):
    @property
    @abstractmethod
    def result_ftype(self):
        """Returns the type of the expression."""
        ...


@dataclass(eq=True, frozen=True)
class Literal(AssemblyExpression):
    """
    Represents the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return type(self.val)

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Variable(AssemblyExpression):
    """
    Represents a logical AST expression for a variable named `name`, which
    will hold a value of type `type`.

    Attributes:
        name: The name of the variable.
        type: The type of the variable.
    """

    name: str
    type: Any

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return self.type

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Allocate(AssemblyTree):
    buffer: Variable

    @property
    def children(self):
        return [self.buffer]

    @classmethod
    def from_children(cls, buffer):
        return cls(buffer)


@dataclass(eq=True, frozen=True)
class Index(AssemblyExpression):
    """
    Represents a logical AST expression for a index named `name`

    Attributes:
        name: The name of the index.
    """

    name: str

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return type(self.name)

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Call(AssemblyExpression, AssemblyTree):
    """
    Represents an expression for calling the function `op` on `args...`.

    Attributes:
        op: The function to call.
        args: The arguments to call on the function.
    """

    op: Literal
    args: tuple[AssemblyNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        arg_types = [arg.result_ftype for arg in self.args]
        return return_type(self.op.val, *arg_types)


@dataclass(eq=True, frozen=True)
class Load(AssemblyExpression, AssemblyTree):
    """
    Represents loading a value from a buffer at given indices.

    Attributes:
        buffer: The buffer to load from.
        indices: The indices to load at (tuple of expressions for
            multidimensional access).
    """

    buffer: Variable
    indices: tuple[
        AssemblyExpression, ...
    ]  # Should be a tuple expression for multidimensional arrays

    @property
    def children(self):
        return [self.buffer, *self.indices]

    @classmethod
    def from_children(cls, buffer, *indices):
        return cls(buffer, indices)

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return element_type(self.buffer.result_ftype)


@dataclass(eq=True, frozen=True)
class Store(AssemblyTree):
    """
    Represents storing a value into a buffer at given indices.

    Attributes:
        buffer: The buffer to store into.
        indices: The indices to store at (tuple of expressions for
            multidimensional access).
        value: The value to store.
    """

    buffer: Variable
    indices: tuple[
        AssemblyExpression, ...
    ]  # Should be a tuple expression for multidimensional arrays
    value: AssemblyExpression

    @property
    def children(self):
        return [self.buffer, *self.indices, self.value]

    @classmethod
    def from_children(cls, buffer, *indices, value):
        return cls(buffer, indices, value)


# All vector instructions assume that the last dimension
# is vectorized.
class AssemblyVectorExpression(AssemblyExpression):
    pass


@dataclass(eq=True, frozen=True)
class VectorCall(AssemblyVectorExpression, AssemblyTree):
    op: Literal
    args: tuple[AssemblyVectorExpression, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        arg_types = [arg.result_ftype for arg in self.args]
        return return_type(self.op.val, *arg_types)


@dataclass(eq=True, frozen=True)
class VectorBroadcast(AssemblyVectorExpression, AssemblyTree):
    value: AssemblyExpression

    @property
    def children(self):
        return [self.value]

    @classmethod
    def from_children(cls, value):
        return cls(value)

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return element_type(self.value.result_ftype)


@dataclass(eq=True, frozen=True)
class VectorGather(AssemblyVectorExpression, AssemblyTree):
    buffer: Variable
    base_offset_idxs: tuple[AssemblyExpression, ...]
    stride: AssemblyExpression

    @property
    def children(self):
        return [self.buffer, self.base_offset_idxs, self.stride]

    @classmethod
    def from_children(cls, buffer, base_offset_idxs, stride):
        return cls(buffer, base_offset_idxs, stride)

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return element_type(self.buffer.result_ftype)


@dataclass(eq=True, frozen=True)
class VectorScatter(AssemblyTree):
    buffer: Variable
    base_offset_idxs: tuple[AssemblyExpression, ...]
    stride: AssemblyExpression
    value: AssemblyVectorExpression

    @property
    def children(self):
        return [self.buffer, self.base_offset_idxs, self.stride, self.value]

    @classmethod
    def from_children(cls, buffer, base_offset_idxs, stride, value):
        return cls(buffer, base_offset_idxs, stride, value)


@dataclass(eq=True, frozen=True)
class Block(AssemblyTree):
    """
    Represents a statement that executes a sequence of statements `bodies...`.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[AssemblyNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.bodies]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)


@dataclass(eq=True, frozen=True)
class ForLoop(AssemblyTree):
    """
    Represents a for loop that iterates over a range of values.

    Attributes:
        lvl: The loop variable.
        start: The starting value of the range.
        end: The ending value of the range.
        body: The body of the loop to execute.
    """

    lvl: Index
    start: AssemblyExpression
    end: AssemblyExpression
    stride: AssemblyExpression
    body: Block

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.lvl, self.start, self.end, self.stride, self.body]


@dataclass(eq=True, frozen=True)
class Function(AssemblyTree):
    """
    Represents a logical AST statement that defines a function `fun` on the
    arguments `args...`.

    Attributes:
        name: The name of the function to define as a literal typed with the
            return type of this function.
        args: The arguments to the function.
        body: The body of the function. If it does not contain a return statement,
            the function returns the value of `body`.
    """

    name: Variable
    args: tuple[Variable, ...]
    body: Block

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.name, *self.args, self.body]

    @classmethod
    def from_children(cls, name, *args, body):
        """Creates a term with the given head and arguments."""
        return cls(name, args, body)


@dataclass(eq=True, frozen=True)
class Return(AssemblyTree):
    """
    Represents a return statement that returns `arg` from the current function.
    Halts execution of the function body.

    Attributes:
        arg: The argument to return.
    """

    arg: AssemblyExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.arg]


@dataclass(eq=True, frozen=True)
class Module(AssemblyTree):
    """
    Represents a group of functions. This is the toplevel translation unit for
    Assembly.

    Attributes:
        funcs: The functions defined in the module.
    """

    funcs: tuple[Function, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.funcs]

    @classmethod
    def from_children(cls, *funcs):
        return cls(funcs)


# TODO: Add cases for index and vectorized nodes
class AssemblyPrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "AssemblyPrinterContext":
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: AssemblyNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return qual_str(value)
            case Variable(name, _):
                return str(name)
            case Call(Literal(_) as lit, args):
                return f"{self(lit)}({', '.join(self(arg) for arg in args)})"
            case Load(buf, idxs):
                return f"load({self(buf)}, {', '.join(self(idx) for idx in idxs)})"
            case Store(buf, idxs, val):
                self.exec(
                    f"{feed}store({self(buf)}, {', '.join(self(idx) for idx in idxs)})"
                )
                return None
            case Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case ForLoop(var, start, end, stride, body):
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(
                    f"{feed}for {var_2} in "
                    f"range({start}, {end}, {stride}):\n{body_code}"
                )
                return None
            case Function(Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case Variable(name, t):
                            arg_decls.append(f"{name}: {qual_str(t)}")
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                self.exec(
                    f"{feed}def {func_name}({', '.join(arg_decls)}) -> "
                    f"{qual_str(return_t)}:\n"
                    f"{body_code}\n"
                )
                return None
            case Return(value):
                self.exec(f"{feed}return {self(value)}")
                return None
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError
