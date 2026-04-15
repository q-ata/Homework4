from .dependency_graph import DependencyGraphNode, DependencyGraphEdge
from .. import nodes as smpl

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
        loop_metadata: Dictionary mapping loop indices to their (start, end, stride)
                      bounds. Can be used to determine if dependencies exist within loop ranges.

    Returns:
        A tuple of DependencyGraphEdge objects representing all detected dependencies
        between the two statements. Empty tuple if no dependencies exist.
    """
    pass
