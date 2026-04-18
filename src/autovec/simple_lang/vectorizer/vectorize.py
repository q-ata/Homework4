from .dependency_graph import (
    DependencyGraphNode,
    DependencyGraphEdge,
    construct_dependency_graph,
    collect_loop_metadata,
    collect_loop_levels
)
from .normalize import normalize
from collections.abc import Callable
from .. import nodes as smpl
from dataclasses import dataclass
import copy
from .dependency_testing import get_siv_form
from ... import symbolic as sym

@dataclass(eq=True, frozen=True)
class SccNode:
    id: int

def codegen(dependency_graph: dict[DependencyGraphNode, set[DependencyGraphEdge]], loop_lvls: list[smpl.Index], loop_metadata: dict[smpl.Index, tuple[int, int, int]], depth: int) -> list[smpl.SimpleLangNode]:
    # if depth >= len(loop_lvls):
    #     return []
    def tarjans(graph: dict[DependencyGraphNode, set[DependencyGraphEdge]]):
            indexes = {}
            low_link = {}
            on_stack = {}
            stack = []
            index = 0
            sccs = []
            def strong_connect(node):
                nonlocal index
                indexes[node] = index
                low_link[node] = index
                index += 1
                stack.append(node)
                on_stack[node] = True

                for target_edge in graph[node]:
                    target = target_edge.target
                    if target not in indexes:
                        strong_connect(target)
                        low_link[node] = min(low_link[node], low_link[target])
                    elif target in on_stack and on_stack[target]:
                        low_link[node] = min(low_link[node], indexes[target])
                
                if low_link[node] == indexes[node]:
                    scc = set()

                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.add(w)
                        if w == node:
                            break
                    sccs.append(scc)

            for node, _ in graph.items():
                if node not in indexes:
                    strong_connect(node)
            return sccs
        
    sccs: list[set[DependencyGraphNode]] = tarjans(dependency_graph)
    def get_min_id(node_set: set[DependencyGraphNode]) -> int:
        node_list = list(node_set)
        min_id = node_list[0].id
        for node in node_list[1:]:
            min_id = min(min_id, node.id)
        return min_id
    sccs.sort(key=get_min_id)
    scc_graph: dict[SccNode, list[SccNode]] = {}
    internal_nodes: dict[SccNode, set[DependencyGraphNode]] = {}
    base_to_scc: dict[DependencyGraphNode, SccNode] = {}
    is_cyclic: dict[SccNode, bool] = {}

    id = 0
    for group in sccs:
        scc_node = SccNode(id)
        id += 1
        scc_graph[scc_node] = []
        internal_nodes[scc_node] = set()
        for base in group:
            base_to_scc[base] = scc_node
            internal_nodes[scc_node].add(base)
        if len(group) == 1:
            # tentative, can be adjusted to true later as a result of self cycles
            is_cyclic[scc_node] = False
        else:
            is_cyclic[scc_node] = True

    for scc_node, internals in internal_nodes.items():
        for internal in internals:
            for target in dependency_graph[internal]:
                scc_target = base_to_scc[target.target]
                if scc_target == scc_node:
                    is_cyclic[scc_node] = True # self cycle, guaranteed cyclic
                else:
                    scc_graph[scc_node].append(scc_target)

    def topo_sort(graph: dict[SccNode, list[SccNode]]) -> list[SccNode]:
        output: list[SccNode] = []
        unvisited: list[SccNode] = list(graph.keys())
        unvisited.sort(key=lambda node: -node.id)
        temp_marked: set[SccNode] = set()
        def visit(node):
            nonlocal output
            if node not in unvisited:
                return
            if node in temp_marked:
                # assert False, "SCC DAG has a cycle?"
                pass
            temp_marked.add(node)
            for target in graph[node]:
                visit(target)
            unvisited.remove(node)
            output = [node] + output
        
        while len(unvisited) > 0:
            element = unvisited[0]
            # print(element)
            visit(element)
        return output
    
    # returns the index in the indices list for where the specified smpl.Index occurs
    # also gives the coefficient and addend for that expreesion.
    # returns None if the supplied smpl.Index does not occur in the indices list
    def find_index_mention(indices: list[smpl.SimpleLangExpression], to_find: smpl.Index) -> tuple[int, int, int] | None:
        for i in range(len(indices)):
            idx = indices[i]
            siv = get_siv_form(idx)
            # print(f"  {siv}, checking against {to_find.name}")
            if siv != None and siv[0] == to_find.name:
                return (i, siv[1], siv[2])
        return None
    
    components = topo_sort(scc_graph)
    already_done = []
    output = []
    for scc_node in components:
        # print(internal_nodes[scc_node])
        if is_cyclic[scc_node]:
            graph_copy = copy.deepcopy(dependency_graph)
            index_to_remove = loop_lvls[depth]
            for node, edges in dependency_graph.items():
                for edge in edges:
                    if edge.loop_lvl == index_to_remove:
                        graph_copy[node].remove(edge)
            for k in already_done:
                del graph_copy[k]
            loop_body = codegen(graph_copy, loop_lvls, loop_metadata, depth + 1)
            loop_meta = loop_metadata[index_to_remove]
            new_loop = smpl.ForLoop(index_to_remove, smpl.Literal(loop_meta[0]), smpl.Literal(loop_meta[1]), smpl.Literal(loop_meta[2]), smpl.Block(tuple(loop_body)))
            output.append(new_loop)
        else:
            internals = list(internal_nodes[scc_node])
            # assert len(internals) == 1, "acyclic scc should have 1 internal"
            stmt: smpl.Store = internals[0].stmt
            already_done.append(internals[0])
            # print(stmt)

            # find which index to vectorize
            success = False
            for loop_index in loop_lvls[depth:]:
                # find if this loop index appears in the statement
                maybe_mention = find_index_mention(stmt.indices, loop_index)
                # print(f"mention: {maybe_mention}, loop index: {loop_index.name}")
                if maybe_mention != None:
                    (i, coefficient, offset) = maybe_mention
                    new_indices = list(stmt.indices)
                    new_indices[i] = smpl.VectorIndex(offset, loop_metadata[loop_index][1] + offset, coefficient)
                    def rw_vectorize(node):
                        siv = get_siv_form(node)
                        if siv == None or siv[0] != loop_index.name:
                            return node
                        return smpl.VectorIndex(siv[2], loop_metadata[loop_index][1] + siv[2], siv[1])
                    rewrite = sym.PreWalk(rw_vectorize)
                    new_rhs = rewrite(stmt.value)
                    output.append(smpl.Store(stmt.buffer, tuple(new_indices), new_rhs))
                    success = True
            if not success:
                output.append(stmt)
    return output

def advanced_vectorization(
    dependency_graph: dict[DependencyGraphNode, set[DependencyGraphEdge]],
    loop_lvls: list[smpl.Index],
    loop_metadata: dict[smpl.Index, tuple[int, int, int]],
) -> list[smpl.SimpleLangNode]:
    """
    TODO: Recursively vectorize loops using the advanced vectorization algorithm by deciding
    which statements can be parallelized based on the dependency graph.

    Args:
        dependency_graph: Data dependencies between operations
        loop_lvls: Loop levels to process from outermost to innermost
        loop_metadata: Loop bounds and step for each level

    Returns:
        List of vectorized statements and sequential loops
    """
    return codegen(dependency_graph, loop_lvls, loop_metadata, 0)

def vectorize(
    simple_lang_ir: smpl.Function,
    dependency_test: Callable[
        [DependencyGraphNode, DependencyGraphNode],
        tuple[DependencyGraphNode, DependencyGraphEdge] | None,
    ],
) -> smpl.Function:
    """
    Vectorize loops in a SimpleLang function using SCC-based dependency analysis.

    For each ForLoop, constructs a dependency graph, collects loop metadata, and applies
    advanced vectorization to transform independent iterations into vector operations.

    Args:
        simple_lang_ir: Function IR containing ForLoop constructs to vectorize
        dependency_test: Callable that checks for data dependencies between statements

    Returns:
        New Function with vectorized loops and preserved non-loop statements
    """
    vectorized_prgm = []

    for stmt in simple_lang_ir.body.bodies:
        if isinstance(stmt, smpl.ForLoop):
            normalized_stmt = normalize(stmt)
            dependency_graph = construct_dependency_graph(
                normalized_stmt, dependency_test
            )
            loop_lvls = collect_loop_levels(normalized_stmt)
            loop_metadata = collect_loop_metadata(normalized_stmt)
            vectorized_loop = advanced_vectorization(
                dependency_graph, loop_lvls, loop_metadata
            )
            vectorized_prgm.extend(vectorized_loop)
        else:
            vectorized_prgm.append(stmt)

    return smpl.Function(
        simple_lang_ir.name, simple_lang_ir.args, smpl.Block(tuple(vectorized_prgm))
    )
