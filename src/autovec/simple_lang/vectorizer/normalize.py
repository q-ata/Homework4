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
    assert isinstance(loop_root.start, smpl.Literal), "start is always constant Literal"
    assert isinstance(loop_root.start.val, int), "start should be int"
    i_start = loop_root.start.val
    assert isinstance(loop_root.end, smpl.Literal), "end is always constant Literal"
    assert isinstance(loop_root.end.val, int), "end should be int"
    i_end = loop_root.end.val
    assert isinstance(loop_root.stride, smpl.Literal), "stride"
    assert isinstance(loop_root.stride.val, int), "stride should be int"
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
    rewrite = sym.PostWalk(rw)
    new_body = rewrite(copy.deepcopy(loop_root.body))
    assert new_body
    new_loop = smpl.ForLoop(old_i,
                            smpl.Literal(0),
                            smpl.Literal(new_upper),
                            smpl.Literal(1),
                            new_body)
    rewrite = sym.PreWalk(rw_simplify)
    new_loop = rewrite(new_loop)
    assert isinstance(new_loop, smpl.ForLoop), "rw_simplify should be the identity for ForLoops"
    return new_loop

# Returns a new block with recursively normalized loop bounds
def recursive_normalize(nodes: list[smpl.SimpleLangNode],
                        old_i: smpl.Index,
                        new_i: smpl.SimpleLangExpression) -> list[smpl.SimpleLangNode]:
    new_nodes = []
    for stmt in nodes:
        match stmt:
            case smpl.ForLoop(_):
                new_loop = normalize(stmt)
                new_loop = recursive_normalize(new_loop.body.children, old_i, new_i)
                new_nodes.append(new_loop)
            case smpl.Block(_):
                new_body = recursive_normalize(stmt.children, old_i, new_i)
                new_nodes.append(smpl.Block.from_children(new_body))
            case smpl.Index(_):
                if stmt == old_i:
                    new_nodes.append(new_i)
                else:
                    new_nodes.append(stmt)
            case smpl.Call(op, args):
                new_args = recursive_normalize([*args], old_i, new_i)
                new_nodes.append(smpl.Call.from_children(op, new_args))
            case smpl.Load(buffer, indices):
                new_indices = recursive_normalize([*indices], old_i, new_i)
                new_nodes.append(smpl.Load.from_children(buffer, new_indices))
            case smpl.Store(buffer, indices, value):
                new_indices = recursive_normalize([*indices], old_i, new_i)
                new_value = recursive_normalize([value], old_i, new_i)[0]
                new_nodes.append(smpl.Store.from_children(buffer, new_indices, value=new_value))
            case smpl.Return(arg):
                new_arg = recursive_normalize([arg], old_i, new_i)[0]
                new_nodes.append(smpl.Return.from_children(new_arg))
            case smpl.Function(_):
                assert False, "Nested function definition"
            case _:
                # Remaining node types may not contain a smpl.Index
                new_nodes.append(stmt)
    assert len(nodes) == len(new_nodes)
    return new_nodes