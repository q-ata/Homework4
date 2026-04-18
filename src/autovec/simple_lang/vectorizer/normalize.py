from typing import TypeVar

from .. import nodes as smpl
from ... import symbolic as sym
import copy
import operator as op

def rw_simplify(expr: smpl.SimpleLangNode) -> smpl.SimpleLangNode:
    match expr:
        # check for constant folding (const + const etc)
        case smpl.Call(smpl.Literal(op.add), (smpl.Literal(a), smpl.Literal(b))) if isinstance(a, int) and isinstance(b, int):
            return smpl.Literal(a + b)
        case smpl.Call(smpl.Literal(op.sub), (smpl.Literal(a), smpl.Literal(b))) if isinstance(a, int) and isinstance(b, int):
            return smpl.Literal(a - b)
        case smpl.Call(smpl.Literal(op.mul), (smpl.Literal(a), smpl.Literal(b))) if isinstance(a, int) and isinstance(b, int):
            return smpl.Literal(a * b)
        case smpl.Call(smpl.Literal(op.add), (smpl.Call(smpl.Literal(op.add), (a, b)), c)):
            return smpl.Call(smpl.Literal(op.add), (a, smpl.Call(smpl.Literal(op.add), (b, c))))
        case smpl.Call(smpl.Literal(op.sub), (a, b)):
            return smpl.Call(smpl.Literal(op.add), (a, smpl.Call(smpl.Literal(op.mul), (smpl.Literal(-1), b))))
        case smpl.Call(smpl.Literal(op.mul), (smpl.Call(smpl.Literal(op.add), (a, b)), c)):
            return smpl.Call(smpl.Literal(op.add), (
                smpl.Call(smpl.Literal(op.mul), (a, c)), smpl.Call(smpl.Literal(op.mul), (b, c))
            ))
        case smpl.Call(smpl.Literal(op.mul), (a, smpl.Call(smpl.Literal(op.add), (b, c)))):
            return smpl.Call(smpl.Literal(op.add), (
                smpl.Call(smpl.Literal(op.mul), (a, b)), smpl.Call(smpl.Literal(op.mul), (a, c))
            ))
        case smpl.Call(smpl.Literal(op.mul), (smpl.Call(smpl.Literal(op.mul), (a, b)), c)):
            return smpl.Call(smpl.Literal(op.mul), (a, smpl.Call(smpl.Literal(op.mul), (b, c))))
        
        # case smpl.Call(smpl.Literal(op.mul), (smpl.Literal(a), smpl.Literal(b))) if isinstance(a, int) and isinstance(b, int):
        #     return smpl.Literal(a*b)
        case _:
            return expr

def rw_simplify2(expr):
    match expr:
        case smpl.Call(smpl.Literal(op.add), (smpl.Literal(a), b)) if isinstance(a, int) and a < 0:
            return smpl.Call(smpl.Literal(op.sub), (b, smpl.Literal(-a)))
        case smpl.Call(smpl.Literal(op.add), (a, smpl.Literal(b))) if isinstance(b, int) and b < 0:
            return smpl.Call(smpl.Literal(op.sub), (a, smpl.Literal(-b)))
        case _:
            return expr
        
def rw_simplify3(expr):
    match expr:
        # check for identities (a + 0 etc)
        case smpl.Call(smpl.Literal(op.add), (a, smpl.Literal(0))):
            return a
        case smpl.Call(smpl.Literal(op.add), (smpl.Literal(0), a)):
            return a
        case smpl.Call(smpl.Literal(op.sub), (a, smpl.Literal(0))):
            return a
        case smpl.Call(smpl.Literal(op.mul), (smpl.Literal(1), a)):
            return a
        case smpl.Call(smpl.Literal(op.mul), (a, smpl.Literal(1))):
            return a
        case _:
            return expr

def default_rewrite(x, y):
    return x if x is not None else y

class FixedPostWalk:
    """
    A rewriter which recursively rewrites the arguments of each node using
    `rw`, then rewrites the resulting node. If all rewriters return `nothing`,
    returns `nothing`.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw):
        self.rw = rw

    def __call__(self, x):
        if isinstance(x, smpl.TermTree):
            args = x.children
            new_args = list(map(self, args))
            if all(arg is None for arg in new_args):
                return self.rw(x)
            y = None
            if x.head() == smpl.Store:
                args = [*map(lambda x1, x2: default_rewrite(x1, x2), new_args, args)]
                y = smpl.Store(buffer=args[0], value=args[-1], indices=tuple(args[1:-1]))
            else:
                y = x.make_term(
                    x.head(), *map(lambda x1, x2: default_rewrite(x1, x2), new_args, args)
                )
            return default_rewrite(self.rw(y), y)  # type: ignore[return-value]
        return self.rw(x)

class FixedPreWalk:
    """
    A rewriter which recursively rewrites each node using `rw`, then rewrites
    the arguments of the resulting node. If all rewriters return `nothing`,
    returns `nothing`.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw):
        self.rw = rw

    def __call__(self, x):
        y = self.rw(x)
        if y is not None:
            if isinstance(y, smpl.TermTree):
                args = y.children
                if y.head() == smpl.Store:
                    store_args = [default_rewrite(self(arg), arg) for arg in args]
                    return smpl.Store(buffer=store_args[0], value=store_args[-1], indices=tuple(store_args[1:-1]))
                return y.make_term(  # type: ignore[return-value]
                    y.head(), *[default_rewrite(self(arg), arg) for arg in args]
                )
            return y
        if isinstance(x, smpl.TermTree):
            args = x.children
            new_args = list(map(self, args))
            if not all(arg is None for arg in new_args):
                if x.head() == smpl.Store:
                    store_args = [*map(lambda x1, x2: default_rewrite(x1, x2), new_args, args)]
                    return smpl.Store(buffer=store_args[0], value=store_args[-1], indices=tuple(store_args[1:-1]))
                return x.make_term(  # type: ignore[return-value]
                    x.head(),
                    *map(lambda x1, x2: default_rewrite(x1, x2), new_args, args),
                )
        return None

def normalize(loop_root: smpl.ForLoop) -> smpl.ForLoop:
    """
    TODO: Perform loop normalization.

    This function should perform recursive rewrite on a loop to normalize the
    loop bounds.

    Args:
        loop_root: The root loop node to normalize.

    Returns:
        A normalized loop node with normalized bounds.
    """
    
    # "For simplicity of implementation, you can assume that the
    # for loop bounds for SimpleLang are always constant Literals."
    # assert isinstance(loop_root.start, smpl.Literal), "start is always constant Literal"
    # assert isinstance(loop_root.start.val, int), "start should be int"
    i_start = loop_root.start.val
    # assert isinstance(loop_root.end, smpl.Literal), "end is always constant Literal"
    # assert isinstance(loop_root.end.val, int), "end should be int"
    i_end = loop_root.end.val
    # assert isinstance(loop_root.stride, smpl.Literal), "stride"
    # assert isinstance(loop_root.stride.val, int), "stride should be int"
    i_stride = loop_root.stride.val

    new_upper = (i_end - i_start) // i_stride
    old_i = smpl.Index(loop_root.lvl.name)
    def make_mul(a, b):
        return smpl.Call(smpl.Literal(op.mul), (a, b))
    def make_add(a, b):
        return smpl.Call(smpl.Literal(op.add), (a, b))
    def make_sub(a, b):
        return smpl.Call(smpl.Literal(op.sub), (a, b))
    new_i = make_add(make_mul(old_i, smpl.Literal(i_stride)), smpl.Literal(i_start))
    
    def rw(node: smpl.SimpleLangNode) -> smpl.SimpleLangNode:
        if node == loop_root.lvl:
            return new_i
        if isinstance(node, smpl.ForLoop):
            # recurse
            return normalize(node)
        return node
    rewrite = FixedPostWalk(rw)
    new_body = rewrite(copy.deepcopy(loop_root.body))
    # assert new_body
    new_loop = smpl.ForLoop(old_i,
                            smpl.Literal(0),
                            smpl.Literal(new_upper),
                            smpl.Literal(1),
                            new_body)
    
    rewrite = FixedPostWalk(rw_simplify)
    new_loop_ = rewrite(new_loop)
    while new_loop_ != new_loop:
        new_loop = new_loop_
        new_loop_ = rewrite(new_loop)

    rewrite = FixedPostWalk(rw_simplify2)
    new_loop_ = rewrite(new_loop)
    while new_loop_ != new_loop:
        new_loop = new_loop_
        new_loop_ = rewrite(new_loop)

    rewrite = FixedPostWalk(rw_simplify3)
    new_loop_ = rewrite(new_loop)
    while new_loop_ != new_loop:
        new_loop = new_loop_
        new_loop_ = rewrite(new_loop)

    # print(new_loop)
    # assert isinstance(new_loop, smpl.ForLoop), "rw_simplify should be the identity for ForLoops"
    return new_loop




# indices=(Call(op=Literal(val=_operator.add),
# args=(Call(op=Literal(val=_operator.mul),
# args=(Index(name='i'),
# Literal(val=1))),
# Literal(val=1))),)
