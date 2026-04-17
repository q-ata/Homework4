from .dependency_graph import DependencyGraphNode, DependencyGraphEdge
from .. import nodes as smpl
import operator
from ... import symbolic as sym

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

def test_strong_siv(i: smpl.SimpleLangExpression, j: smpl.SimpleLangExpression) -> tuple[smpl.Index, int] | None:
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
    return (smpl.Index(i_name), int(dist))

def test_weak_zero_siv(i: smpl.SimpleLangExpression, j: smpl.SimpleLangExpression) -> tuple[smpl.Index, int] | None:
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
        return None # weak zero not applicable: other index is not literal
    
    (name, coeff, offset) = strong_tuple
    dist = (lit - offset) / coeff
    if not dist.is_integer():
        return None
    return (smpl.Index(name), int(dist))
    

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
    
    def reorder_distances(distances: list[tuple[smpl.Index, int]], a_before_b) -> list[tuple[smpl.Index, int]]:
        is_ziv = True
        for (i, _) in distances:
            if i.name != "ziv":
                is_ziv = False
                break
        if is_ziv:
            return []
        new_dists = []
        for level in loop_lvls:
            found = False
            for (i, dist) in distances:
                if level == i:
                    new_dists.append((i, dist))
                    found = True
                    break
            if not found and not a_before_b:
                # There is a loop whose index is not referenced.
                # Artificially insert a "<" direction so the later code that
                # checks plausibility will succeed.
                new_dists.append((level, 1))
        return new_dists
    
    ziv_index = smpl.Index("ziv")
    
    # Need to check for:
    # RAW from s1 to s2
    # RAW from s2 to s1
    # WAR from s2 to s1
    # WAR from s1 to s2
    pairs_to_check = []
    for load in loads2:
        if s1.stmt.buffer != load.buffer:
            continue
        # True means the second index's instruction comes lexically AFTER the first.
        # for self cycle, the write comes after the read
        # otherwise, s1 comes before s2 which means the load comes after
        pairs_to_check.append((s1.stmt.indices, load.indices, s1.id != s2.id, s1, s2))
        pairs_to_check.append((load.indices, s1.stmt.indices, s1.id == s2.id, s2, s1))
    
    # self cycles were already covered by the first loop's appends
    if s1.id != s2.id:
        for load in loads1:
            if s2.stmt.buffer != load.buffer:
                continue
            # guaranteed s2 store comes lexically after the load
            pairs_to_check.append((s2.stmt.indices, load.indices, False, s2, s1))
            # check for WAR from s1 to s2, s1 comes lexically before
            pairs_to_check.append((load.indices, s2.stmt.indices, True, s1, s2))

    # WAW
    if s1.stmt.buffer == s2.stmt.buffer:
        pairs_to_check.append((s1.stmt.indices, s2.stmt.indices, s1.id != s2.id, s1, s2))
        if s1.id != s2.id:
            pairs_to_check.append((s2.stmt.indices, s1.stmt.indices, False, s2, s1))

    # check for dep from a to b
    for (indices1, indices2, a_before_b, node_a, node_b) in pairs_to_check:
        # print(f"index 1: {indices1}")
        # print(f"index 2: {indices2}")
        # try to construct a plausible distance vector.
        # if the direction vector is all =, its only valid if the dependency is ziv or a_before_b is TRUE
        distances = []
        found_dep = True
        for (index1, index2) in zip(indices1, indices2):
            maybe_dep = test_ziv(index1, index2)
            if maybe_dep:
                distances.append(maybe_dep)
            else:
                maybe_dep = test_strong_siv(index1, index2)
                if maybe_dep:
                    distances.append(maybe_dep)
                else:
                    maybe_dep = test_weak_zero_siv(index1, index2)
                    if maybe_dep:
                        distances.append(maybe_dep)
                    else:
                        # couldn't find a conflict with all tests, assume no dependency
                        found_dep = False
                        break
        if not found_dep:
            continue

        # there is potential for conflict in all the indices
        # reorder the indices into outer -> inner loop order
        # if the new distances is empty, the indices were entire ziv
        distances = reorder_distances(distances, a_before_b)
        # print(distances)
        # if the distance vector is plausible AND a_before_b, a dependency exists
        # if not a_before_b a dependency exists iff (the distance vector is plausible and non zero) or its empty, meaning ziv
        
        # new distances being empty means entirely ziv, and there is conflict
        if len(distances) == 0:
            deps.append(DependencyGraphEdge(ziv_index, node_a, node_b))
        else:
            plausible = True
            non_zero = False
            carry = distances[-1][0]
            for (idx, dist) in distances:
                if dist < 0 and not non_zero:
                    # the first non zero distance is negative, implausible
                    plausible = False
                    break
                if dist > 0 and not non_zero:
                    # still need to check loop bounds, cant break yet
                    non_zero = True
                    carry = idx
                if dist != 0 and abs(dist) >= loop_metadata[idx][1]:
                    # the distance exceeds loop bounds, implausible
                    plausible = False
                    break
            if plausible and (a_before_b or non_zero):
                deps.append(DependencyGraphEdge(carry, node_a, node_b))
    return tuple(deps)