from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..symbolic import ScopedDict, fisinstance
from . import nodes as asm


class SimpleLangAssemblyInterpreterKernel:
    """
    A kernel for interpreting SimpleLang Assembly code.
    This is a simple interpreter that executes the assembly code.
    """

    def __init__(self, ctx, func_n, ret_t):
        self.ctx = ctx
        self.func = asm.Variable(func_n, ret_t)

    def __call__(self, *args):
        args_i = (asm.Literal(arg) for arg in args)
        return self.ctx(asm.Call(self.func, args_i))


class SimpleLangAssemblyInterpreterModule:
    """
    A class to represent an interpreted module of SimpleLang Assembly.
    """

    def __init__(self, ctx, kernels):
        self.ctx = ctx
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass(eq=True)
class HaltState:
    """
    A class to represent the halt state of an assembly program.
    This is used to indicate whether we should break or return, and
    what the return value is if applicable.
    """

    should_halt: bool = False
    return_value: Any = None


class SimpleLangAssemblyInterpreter:
    """
    An interpreter for SimpleLang Assembly.
    """

    def __init__(
        self,
        bindings=None,
        types=None,
        loop_state=None,
        function_state=None,
    ):
        if bindings is None:
            bindings = ScopedDict()
        if types is None:
            types = ScopedDict()
        self.bindings = bindings
        self.types = types
        self.loop_state = loop_state
        self.function_state = function_state

    def scope(
        self,
        bindings=None,
        types=None,
        loop_state=None,
        function_state=None,
    ):
        """
        Create a new scope for the interpreter.
        This allows for nested scopes and variable shadowing.
        """
        if bindings is None:
            bindings = self.bindings.scope()
        if types is None:
            types = self.types.scope()
        if loop_state is None:
            loop_state = self.loop_state
        if function_state is None:
            function_state = self.function_state
        return SimpleLangAssemblyInterpreter(
            bindings=bindings,
            types=types,
            loop_state=loop_state,
            function_state=function_state,
        )

    def should_halt(self):
        """
        Check if the interpreter should halt execution.
        This is used to stop execution in loops or when a return
        statement is encountered.
        """
        return (
            self.loop_state
            and self.loop_state.should_halt
            or self.function_state
            and self.function_state.should_halt
        )

    def __call__(self, prgm: asm.AssemblyNode):
        """
        Run the program.
        """
        match prgm:
            case asm.Literal(value):
                return value
            case asm.Variable(var_n, var_t):
                if var_n in self.types:
                    def_t = self.types[var_n]
                    if def_t != var_t:
                        raise TypeError(
                            f"Variable '{var_n}' is declared as type {def_t}, "
                            f"but used as type {var_t}."
                        )
                if var_n in self.bindings:
                    return self.bindings[var_n]
                raise KeyError(
                    f"Variable '{var_n}' is not defined in the current context."
                )
            case asm.Call(f, args):
                f_e = self(f)
                args_e = [self(arg) for arg in args]
                return f_e(*args_e)
            case asm.Load(buf, indices):
                buf_e = self(buf)
                indices_e = [self(idx) for idx in indices]
                return buf_e.load(indices_e)
            case asm.Store(buf, indices, val):
                buf_e = self(buf)
                indices_e = [self(idx) for idx in indices]
                val_e = self(val)
                buf_e.store(indices_e, val_e)
                return None
            case asm.Block(bodies):
                for body in bodies:
                    if self.should_halt():
                        break
                    self(body)
                return None
            case asm.ForLoop(
                asm.Variable(var_n, var_t) as var, start, end, stride, body
            ):
                start_e = self(start)
                end_e = self(end)
                stride_e = self(stride)
                if not isinstance(start_e, var_t):
                    raise TypeError(
                        f"Start value {start_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                ctx_2 = self.scope(loop_state=HaltState())
                var_e = start_e
                while var_e < end_e:
                    if ctx_2.should_halt():
                        break

                    ctx_3 = self.scope()

                    # Updating the loop variable
                    self.bindings[var_n] = var_e
                    self.types[var_n] = var_t

                    ctx_3(body)
                    var_e = type(var_e)(var_e + stride_e)  # type: ignore[call-arg,operator]

                return None
            case asm.Function(asm.Variable(func_n, ret_t), args, body):

                def my_func(*args_e):
                    ctx_2 = self.scope(function_state=HaltState())
                    if len(args_e) != len(args):
                        raise ValueError(
                            f"Function '{func_n}' expects {len(args)} arguments, "
                            f"but got {len(args_e)}."
                        )
                    for arg, arg_e in zip(args, args_e, strict=False):
                        match arg:
                            case asm.Variable(arg_n, arg_t):
                                if not fisinstance(arg_e, arg_t):
                                    raise TypeError(
                                        f"Argument '{arg_n}' is expected to be of type "
                                        f"{arg_t}, but got {type(arg_e)}."
                                    )
                                ctx_2.bindings[arg_n] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if ctx_2.function_state.should_halt:
                        ret_e = ctx_2.function_state.return_value
                        if ret_e.ftype != ret_t:
                            raise TypeError(
                                f"Return value {ret_e} is not of type {ret_t} "
                                f"for function '{func_n}'."
                            )
                        return ret_e
                    raise ValueError(
                        f"Function '{func_n}' did not return a value, "
                        f"but expected type {ret_t}."
                    )

                self.bindings[func_n] = my_func
                return None
            case asm.Return(value):
                self.function_state.return_value = self(value)
                self.function_state.should_halt = True
                return None
            case asm.Module(funcs):
                for func in funcs:
                    self(func)
                kernels = {}
                for func in funcs:
                    match func:
                        case asm.Function(asm.Variable(func_n, ret_t), args, _):
                            kernel = SimpleLangAssemblyInterpreterKernel(self, func_n, ret_t)
                            kernels[func_n] = kernel
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized function definition: {func}"
                            )
                return SimpleLangAssemblyInterpreterModule(self, kernels)
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
