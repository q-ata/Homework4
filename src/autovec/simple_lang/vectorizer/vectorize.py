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
    pass


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
