from .dependency_graph import DependencyGraphNode, DependencyGraphEdge
from .. import nodes as smpl
import operator
from ... import symbolic as sym
import copy
import math
from enum import Enum

# returns None if no dependency could be found (when the constants don't match or ZIV is unapplicable)
# Otherwise, returns the Index that carries the dep (N/A for ZIV) and the distance (default 0 for ZIV)
def test_ziv(i: smpl.SimpleLangExpression, j: smpl.SimpleLangExpression) -> tuple[smpl.Index, int] | None:
    # ziv requires literals for both indices, this is post simplify pass
    if (not isinstance(i, smpl.Literal)) or (not isinstance(i.val, int)) or (not isinstance(j, smpl.Literal)) or (not isinstance(j.val, int)):
        return None
    if i.val != j.val:
        return None
    return (smpl.Index("ziv"), 0)

def get_siv_form(i):
    match i:
        # aI+c
        case smpl.Call(smpl.Literal(operator.add), (smpl.Call(smpl.Literal(operator.mul), (a, b)), smpl.Literal(c))) if isinstance(c, int):
            match (a, b):
                # aI+c
                case (smpl.Index(name), smpl.Literal(coeff)) if isinstance(coeff, int):
                    return (name, coeff, c)
                # Ia+c
                case (smpl.Literal(coeff), smpl.Index(name)) if isinstance(coeff, int):
                    return (name, coeff, c)
                case _:
                    return None
        # c+aI
        case smpl.Call(smpl.Literal(operator.add), (smpl.Literal(c), smpl.Call(smpl.Literal(operator.mul), (a, b)))) if isinstance(c, int):
            match (a, b):
                # c+aI
                case (smpl.Index(name), smpl.Literal(coeff)) if isinstance(coeff, int):
                    return (name, coeff, c)
                # c+Ia
                case (smpl.Literal(coeff), smpl.Index(name)) if isinstance(coeff, int):
                    return (name, coeff, c)
                case _:
                    return None
        # I+c
        case smpl.Call(smpl.Literal(operator.add), (smpl.Index(name), smpl.Literal(addend))) if isinstance(addend, int):
            return (name, 1, addend)
        # c+I
        case smpl.Call(smpl.Literal(operator.add), (smpl.Literal(addend), smpl.Index(name))) if isinstance(addend, int):
            return (name, 1, addend)
        # aI-c
        case smpl.Call(smpl.Literal(operator.sub), (smpl.Call(smpl.Literal(operator.mul), (a, b)), smpl.Literal(c))) if isinstance(c, int):
            match (a, b):
                # aI-c
                case (smpl.Index(name), smpl.Literal(coeff)) if isinstance(coeff, int):
                    return (name, coeff, -c)
                # Ia-c
                case (smpl.Literal(coeff), smpl.Index(name)) if isinstance(coeff, int):
                    return (name, coeff, -c)
                case _:
                    return None
        # c-aI
        case smpl.Call(smpl.Literal(operator.sub), (smpl.Literal(c), smpl.Call(smpl.Literal(operator.mul), (a, b)))) if isinstance(c, int):
            match (a, b):
                # c-aI
                case (smpl.Index(name), smpl.Literal(coeff)) if isinstance(coeff, int):
                    return (name, -coeff, c)
                # c-Ia
                case (smpl.Literal(coeff), smpl.Index(name)) if isinstance(coeff, int):
                    return (name, -coeff, c)
                case _:
                    return None
        # I-c
        case smpl.Call(smpl.Literal(operator.sub), (smpl.Index(name), smpl.Literal(addend))) if isinstance(addend, int):
            return (name, 1, -addend)
        # c-I
        case smpl.Call(smpl.Literal(operator.sub), (smpl.Literal(addend), smpl.Index(name))) if isinstance(addend, int):
            return (name, -1, addend)
        # aI
        case smpl.Call(smpl.Literal(operator.mul), (smpl.Index(name), smpl.Literal(coeff))) if isinstance(coeff, int):
            return (name, coeff, 0)
        # Ia
        case smpl.Call(smpl.Literal(operator.mul), (smpl.Literal(coeff), smpl.Index(name))) if isinstance(coeff, int):
            return (name, coeff, 0)
        # I
        case smpl.Index(name):
            return (name, 1, 0)
        case _:
            return None

class Direction(Enum):
    LT = "<"
    GT = ">"
    EQ = "="
    ALL = "*"

    @staticmethod
    def from_distance(dist: int):
        if dist < 0:
            return Direction.GT
        if dist == 0:
            return Direction.EQ
        return Direction.LT

def test_strong_siv(i: smpl.SimpleLangExpression, j: smpl.SimpleLangExpression, loop_meta) -> list[tuple[smpl.Index, Direction]] | None:
    # check for correct form
    (i_, j_) = (get_siv_form(i), get_siv_form(j))
    if not i_ or not j_:
        return None
    (i_name, i_coeff, i_addend) = i_
    (j_name, j_coeff, j_addend) = j_
    if i_name != j_name or i_coeff != j_coeff:
        return None
    
    dist = (i_addend - j_addend) / i_coeff
    if not dist.is_integer():
        return None
    # print(dist)
    upper_bound = loop_meta[smpl.Index(i_name)][1]
    if abs(dist) < upper_bound:
        return [(smpl.Index(i_name), Direction.from_distance(int(dist)))]
    return None # exceeds loop bounds

def test_weak_zero_siv(i: smpl.SimpleLangExpression, j: smpl.SimpleLangExpression, loop_meta) -> list[tuple[smpl.Index, Direction]] | None:
    i_ = get_siv_form(i)
    j_= get_siv_form(j)
    if (i_ == None and j_ == None) or (i_ != None and j_ != None):
        # weak zero is not applicable
        return None
    
    # guaranteed exactly one of i_ and j_ is None
    def get_literal_value(i):
        if not isinstance(i, smpl.Literal):
            return None
        if not isinstance(i.val, int):
            return None
        return i.val
    strong_tuple = None
    lit = None
    if i_ == None:
        strong_tuple = j_
        lit = get_literal_value(i)
    else:
        strong_tuple = i_
        lit = get_literal_value(j)
    
    if strong_tuple == None or lit == None:
        # print("not applicable:")
        # print(i)
        # print(j)
        return None # weak zero not applicable: other index is not literal
    
    (name, coeff, offset) = strong_tuple
    match_point = (lit - offset) / coeff
    if not match_point.is_integer():
        return None
    
    upper_bound = loop_meta[smpl.Index(name)][1]
    if match_point < 0 or match_point >= upper_bound:
        return None
    possible = []
    if match_point < upper_bound:
        possible.append((smpl.Index(name), Direction.EQ))
    if match_point < upper_bound - 1:
        possible.append((smpl.Index(name), Direction.GT))
    if match_point > 0:
        possible.append((smpl.Index(name), Direction.LT))
    # assert len(possible) > 0, "no possible?"
    return possible
    

def dependency_test(
    stmt1: DependencyGraphNode,
    stmt2: DependencyGraphNode,
    loop_lvls: list[smpl.Index],
    loop_metadata: dict[smpl.Index, tuple[int, int, int]],
) -> tuple[DependencyGraphEdge, ...]:
    """
    TODO: Perform dependency analysis between stmt1 and stmt2.

    This function should implement dependency testing for Flow, Anti and Output dependencies.
    Implement tests for ZIV, Strong SIV and Weak Zero SIV.

    Args:
        stmt1: Statement 1
        stmt2: Statement 2
        loop_lvls: Loop levels ordered from outermost to innermost
        loop_metadata: Dictionary mapping loop indices to their (start, end, stride)
                      bounds. Can be used to determine if dependencies exist within loop ranges.

    Returns:
        A tuple of DependencyGraphEdge objects representing all detected dependencies
        between the two statements. Empty tuple if no dependencies exist.
    """
    
    deps = []

    s1 = stmt1
    s2 = stmt2
    # ensure s1 is lexicographically before s2
    if s1.id > s2.id:
        temp = s1
        s1 = s2
        s2 = temp
        
    loads1: list[smpl.Load] = []
    def rw1(node: smpl.SimpleLangNode) -> smpl.SimpleLangNode:
        if isinstance(node, smpl.Load):
            loads1.append(node)
        return node
    rewrite = sym.PreWalk(rw1)
    rewrite(s1.stmt.value)
    
    loads2: list[smpl.Load] = []
    def rw2(node: smpl.SimpleLangNode) -> smpl.SimpleLangNode:
        if isinstance(node, smpl.Load):
            loads2.append(node)
        return node
    rewrite = sym.PreWalk(rw2)
    rewrite(s2.stmt.value)

    pairs_to_check = []
    for load in loads2:
        if s1.stmt.buffer != load.buffer:
            continue
        # True means the second index's instruction comes lexically AFTER the first.
        # for self cycle, the write comes after the read
        # otherwise, s1 comes before s2 which means the load comes after
        # the last tuple component signals whether s2 is lexically before s1
        # for s1.id == s2.id, the read comes first
        pairs_to_check.append((s1.stmt.indices, load.indices, s1, s2, False, s1.id != s2.id))
        pairs_to_check.append((load.indices, s1.stmt.indices, s2, s1, False, s1.id == s2.id))
    
    # self cycles were already covered by the first loop's appends
    if s1.id != s2.id:
        for load in loads1:
            if s2.stmt.buffer != load.buffer:
                continue
            # guaranteed s2 store comes lexically after the load
            pairs_to_check.append((s2.stmt.indices, load.indices, s2, s1, False, False))
            # check for WAR from s1 to s2, s1 comes lexically before
            pairs_to_check.append((load.indices, s2.stmt.indices, s1, s2, False, True))

    # WAW
    if s1.stmt.buffer == s2.stmt.buffer:
        pairs_to_check.append((s1.stmt.indices, s2.stmt.indices, s1, s2, s1.id == s2.id, True))
        if s1.id != s2.id:
            pairs_to_check.append((s2.stmt.indices, s1.stmt.indices, s2, s1, False))

    index_to_pos = {}
    counter = 0
    for i in loop_lvls:
        index_to_pos[i] = counter
        counter += 1

    def merge_vector_sets(dirs, cur_vecs):
        new_vecs = []
        for (index, dir) in dirs:
            for vec in cur_vecs:
                vec_copy = copy.deepcopy(vec)
                vec_copy[index_to_pos[index]] = dir
                new_vecs.append(vec_copy)
        return new_vecs

    def test_dependence(indices1: list[smpl.SimpleLangExpression], indices2: list[smpl.SimpleLangExpression], a_equals_b):
        subscripts = [*zip(indices1, indices2)]
        # special case: all indices are ziv
        is_ziv = True
        for (index1, index2) in subscripts:
            # test_separable
            maybe_dep = test_ziv(index1, index2)
            if maybe_dep == None:
                is_ziv = False
                break
        if is_ziv:
            return "ziv"
        
        # indices 1 and 2 refer to the same access, dont create a dep
        if a_equals_b:
            return []
        
        vecs = [[Direction.ALL] * len(loop_lvls)]
        for (index1, index2) in subscripts:
            # test_separable()
            maybe_dep = test_ziv(index1, index2)
            if maybe_dep:
                continue # ignore ziv
            maybe_dep = test_strong_siv(index1, index2, loop_metadata)
            if maybe_dep != None:
                vecs = merge_vector_sets(maybe_dep, vecs)
            else:
                maybe_dep = test_weak_zero_siv(index1, index2, loop_metadata)
                # print(f"weak zero: {maybe_dep}")
                if maybe_dep != None:
                    vecs = merge_vector_sets(maybe_dep, vecs)
                else:
                    # no dependency, indices can't conflict
                    return []
            # print(f"updated vecs: {vecs}")
        return vecs
            
    def is_plausible(vec):
        for dir in vec:
            if dir == Direction.LT:
                return True
            if dir == Direction.GT:
                return False
        return True # all EQ
    
    def find_carry(vec):
        # print(vec)
        for i in range(len(vec)):
            el = vec[i]
            if el == Direction.LT:
                return loop_lvls[i]
        # all eq
        return None
    
    def discharge_stars(prefix, vec, start_from, accum):
        if start_from >= len(vec):
            accum.append(prefix)
            return
        dir = vec[start_from]
        if dir == Direction.ALL:
            discharge_stars(prefix + [Direction.LT], vec, start_from + 1, accum)
            discharge_stars(prefix + [Direction.EQ], vec, start_from + 1, accum)
            discharge_stars(prefix + [Direction.GT], vec, start_from + 1, accum)
        else:
            discharge_stars(prefix + [dir], vec, start_from + 1, accum)

    def find_some_index(indices1, indices2):
        for (i, j) in [*zip(indices1, indices2)]:
            siv = get_siv_form(i)
            if siv != None:
                return siv[0]
            siv = get_siv_form(j)
            if siv != None:
                return siv[0]
        return None
    
    def is_loop_independent(vec):
        for dir in vec:
            if dir != Direction.EQ:
                return False
        return True

    deps: list[DependencyGraphEdge] = []
    for (indices1, indices2, node_a, node_b, a_equals_b, a_before_b) in pairs_to_check:
        # print(f"indices1: {indices1}")
        # print(f"indices2: {indices2}")
        vecs = test_dependence(indices1, indices2, a_equals_b)
        if vecs == "ziv":
            deps.append(DependencyGraphEdge(smpl.Index("ziv"), node_a, node_b))
            continue

        for vec in vecs:
            concrete_vecs = []
            discharge_stars([], vec, 0, concrete_vecs)
            # print(concrete_vecs)
            for vec_ in concrete_vecs:
                # print(vec_)
                if (not a_before_b or node_a == node_b) and is_loop_independent(vec_):
                    continue
                # print(f"{vec_} {a_before_b}")
                if is_plausible(vec_):
                    carry = find_carry(vec_)
                    if carry == None:
                        c = find_some_index(indices1, indices2)
                        if c == None:
                            c = "ziv"
                        carry = smpl.Index(c)
                    # print(carry.name)
                    deps.append(DependencyGraphEdge(carry, node_a, node_b))

    return tuple(deps)