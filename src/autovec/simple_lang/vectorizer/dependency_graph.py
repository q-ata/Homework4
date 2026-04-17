from ..nodes import Store, Index, ForLoop
from collections.abc import Callable
from dataclasses import dataclass


class DependencyGraphNode:
    """
    A dependency graph node. Each node has a unique ID and contains
    the statement that it represents.
    """

    # static variable used to assign unique ID to dependency
    # graph node. This is used to simplify tests.
    count = 0

    def __init__(self, stmt: Store):
        DependencyGraphNode.count += 1
        self.id = DependencyGraphNode.count
        self.unique_id = f"S{self.id}"
        self.stmt = stmt

    def __eq__(self, other):
        return self.stmt == other.stmt

    def __hash__(self):
        return hash(self.stmt)


@dataclass(eq=True, frozen=True)
class DependencyGraphEdge:
    """
    A dependency graph edge from source to destination, the loop_lvl
    field allows us to uniquely identify the loop within which this
    dependency exists.
    """

    loop_lvl: Index
    source: DependencyGraphNode
    target: DependencyGraphNode


def _construct_empty_dependency_graph(loop_root: ForLoop):
    """
    Helper function to create nodes in the dependency graph
    """
    dependency_graph: dict[DependencyGraphNode, set[DependencyGraphEdge]] = {}

    # Generate dependency graph nodes recursively.
    for stmt in loop_root.body.bodies:
        if isinstance(stmt, Store):
            dependency_graph[DependencyGraphNode(stmt)] = set()
        else:
            dependency_graph.update(_construct_empty_dependency_graph(stmt))

    return dependency_graph


def collect_loop_metadata(loop_root: ForLoop) -> dict[Index, tuple[int, int, int]]:
    """
    Helper function to collect loop metadata
    """
    loop_metadata: dict[Index, tuple[int, int, int]] = {}

    loop_metadata[loop_root.lvl] = (loop_root.start.val, loop_root.end.val, loop_root.stride.val)
    for stmt in loop_root.body.bodies:
        if isinstance(stmt, ForLoop):
            loop_metadata.update(collect_loop_metadata(stmt))

    return loop_metadata

def collect_loop_levels(loop_root: ForLoop) -> list[Index]:
    """
    Recursively collects indexes of all loops. To simplify the implementation
    we assume that all loops are nested within each other.
    """
    loop_lvls = [loop_root.lvl]

    for stmt in loop_root.body.bodies:
        if isinstance(stmt, ForLoop):
            loop_lvls.extend(collect_loop_levels(stmt))

    return loop_lvls

def construct_dependency_graph(
    loop_root: ForLoop,
    dependency_test: Callable[
        [DependencyGraphNode, DependencyGraphNode, list[Index], dict[Index, tuple[int, int, int]]],
        DependencyGraphEdge | None,
    ],
) -> dict[DependencyGraphNode, set[DependencyGraphEdge]]:
    """
    Constructs a dependency graph for a loop.
    Returns the adjacency list for the dependency graph.
    """

    # Generate dependency graph nodes recursively.
    dependency_graph = _construct_empty_dependency_graph(loop_root)

    # Collect loop metadata
    loop_metadata = collect_loop_metadata(loop_root)
    loop_lvls = collect_loop_levels(loop_root)

    # Update the dependency graph to include dependency edges
    # associated with the loop_root.
    dependency_nodes = list(dependency_graph.keys())
    for stmt1 in dependency_nodes:
        for stmt2 in dependency_nodes:
            dependency_edges = dependency_test(stmt1, stmt2, loop_lvls, loop_metadata)
            for edge in dependency_edges:
                dependency_graph[edge.source].add(edge)

    return dependency_graph
