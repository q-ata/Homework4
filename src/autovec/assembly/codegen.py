import logging
import operator

from ..codegen import (
    CContext,
    CKernel,
    CModule,
    c_function_call,
    c_literal,
    c_type,
    init_shared_lib,
)
from ..util import config
from . import nodes as asm

logger = logging.getLogger(__name__)

op_to_vector_map = {
    operator.add: "_mm512_add_pd",
    operator.sub: "_mm512_sub_pd",
    operator.mul: "_mm512_mul_pd",
}


class UnsupportedOperation(Exception):
    pass


class SimpleLangAssembly2CGenerator:
    def __call__(self, prgm: asm.AssemblyNode):
        ctx = SimpleLangAssembly2CContext()
        ctx(prgm)
        return ctx.emit_file()


class SimpleLangAssembly2CCompiler:
    """
    A class to compile and run SimpleLangAssembly.
    """

    def __init__(self, cc=None, cflags=None, shared_cflags=None):
        if cc is None:
            cc = config.get("cc")
        if cflags is None:
            cflags = config.get("cflags").split()
        if shared_cflags is None:
            shared_cflags = config.get("shared_cflags").split()
        self.cc = cc
        self.cflags = cflags
        self.shared_cflags = shared_cflags

    def __call__(self, prgm):
        ctx = SimpleLangAssembly2CContext()

        # Header for vector support
        ctx.add_header("#include <immintrin.h>")

        ctx(prgm)

        c_code = ctx.emit_file()
        c_globals = ctx.emit_globals()
        logger.info(f"Compiling C code:\n{c_code}")
        lib = init_shared_lib(
            c_code=c_code,
            globals=c_globals,
            cc=self.cc,
            cflags=(*self.cflags, *self.shared_cflags),
        )
        kernels = {}
        if prgm.head() != asm.Module:
            raise ValueError(
                "SimpleLang2CCompiler expects a Module as the head of the program, "
                f"got {type(prgm.head())}"
            )
        for func in prgm.funcs:
            match func:
                case asm.Function(asm.Variable(func_name, return_t), args, _):
                    arg_ts = [arg.result_ftype for arg in args]
                    kern = CKernel(getattr(lib, func_name), return_t, arg_ts)
                    kernels[func_name] = kern
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return CModule(lib, kernels)


class SimpleLangAssembly2CContext(CContext):
    """
    A class to represent a C environment.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def cache(self, name, val):
        if isinstance(val, asm.Literal | asm.Variable):
            return val
        var_n = self.freshen(name)
        var_t = val.result_ftype
        var_t_code = self.ctype_name(c_type(var_t))
        self.exec(f"{self.feed}{var_t_code} {var_n} = {self(val)};")
        return asm.Variable(var_n, var_t)

    # TODO: add case for index variable
    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        """
        lower the program to C code.
        """
        match prgm:
            case asm.Literal(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case asm.Variable(name, t):
                return name
            case asm.Index(name):
                return name
            case asm.Call(f, args):
                assert isinstance(f, asm.Literal)
                return c_function_call(f.val, self, *args)
            case asm.Load(buf, indices):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_load(self, buf, indices)
            case asm.Allocate(buf):
                var_t = self.ctype_name(c_type(buf.type))
                alloc_str = buf.type.c_alloc(
                    self, [asm.Literal(dim) for dim in buf.type.shape]
                )
                self.exec(f"{self.feed}{var_t} {buf.name} = {alloc_str};")
                return None
            case asm.Store(buf, indices, val):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_store(self, buf, indices, val)
            case asm.VectorCall(op, args):
                if op.val not in op_to_vector_map:
                    raise UnsupportedOperation(f"{op.val} is unsupported by codegen.")
                return f"{op_to_vector_map[op.val]}({', '.join([self(arg) for arg in args])})"
            case asm.VectorBroadcast(value):
                return f"_mm512_set1_pd({self(value)})"
            case asm.VectorGather(buf, base_offset, stride):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_vecgather(self, buf, base_offset, stride)
            case asm.VectorScatter(buf, base_offset, stride, val):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_vecscatter(
                    self, buf, base_offset, stride, val
                )
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(var, start, end, stride, body):
                var_t = "int64_t"
                var_2 = var.name
                start = self(start)
                end = self(end)
                stride = self(stride)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.types[var.name] = var.result_ftype
                body_code = ctx_2.emit()
                self.exec(
                    f"{feed}for ({var_t} {var_2} = {start}; "
                    f"{var_2} < {end}; {var_2}+={stride}) {{\n"
                    f"{body_code}"
                    f"\n{feed}}}"
                )
                return None
            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t):
                            t_name = self.ctype_name(c_type(t))
                            arg_decls.append(f"{t_name} {name}")
                            ctx_2.types[name] = t
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                return_t_name = self.ctype_name(c_type(return_t))
                feed = self.feed
                self.exec(
                    f"{feed}{return_t_name} {func_name}({', '.join(arg_decls)}) {{\n"
                    f"{body_code}\n"
                    f"{feed}}}"
                )
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value};")
                return None
            case asm.Module(funcs):
                for func in funcs:
                    if not isinstance(func, asm.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
