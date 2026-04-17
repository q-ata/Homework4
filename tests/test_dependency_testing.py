import pytest
from autovec.simple_lang.vectorizer.dependency_graph import (
    construct_dependency_graph,
    DependencyGraphNode,
)
from autovec.simple_lang.vectorizer.dependency_testing import dependency_test
from autovec.simple_lang.parser import SimpleLangParser


@pytest.mark.parametrize(
    "input_prgm, expected_dependency_graph",
    [
        # (
        #     """
        #     function vectorizable1(A[16]) -> [16]:
        #         for i in range(0,16,1)
        #             A[i] = 0    # S1
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": []},
        # ),
        # (
        #     """
        #     function unvectorizable1(A[16]) -> [16]:
        #         for i in range(0,15,1)
        #             A[i+1] = A[i] + 12  # S1
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": [("S1", "i")]},
        # ),
        # (
        #     """
        #     function unvectorizable2(A[16], F[16]) -> [16]:
        #         for i in range(0,15,1)
        #             A[i+1] = F[i]   # S1
        #             F[i+1] = A[i]   # S2
        #         end
        #         return F
        #     end
        #     """,
        #     {"S1": [("S2", "i")], "S2": [("S1", "i")]},
        # ),
        # (
        #     """
        #     function loop_splitting1(A[16], B[16]) -> [16]:
        #         for i in range(1,16,1)
        #             A[i] = B[i] + 1     # S1
        #             B[i-1] = A[i] - 5   # S2
        #         end
        #         return B
        #     end
        #     """,
        #     {"S1": [("S2", "i")], "S2": []},
        # ),
        # (
        #     """
        #     function simple_vectorization_algo1(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
        #         for i in range(0,15,1)
        #             D[i] = A[i] + 4     # S1
        #             A[i+1] = B[i] + 6   # S2
        #             Y[i] = X[i] + D[i]  # S3
        #             X[i+1] = Y[i] + 9   # S4
        #         end
        #         return X
        #     end
        #     """,
        #     {
        #         "S1": [("S3", "i")],
        #         "S2": [("S1", "i")],
        #         "S3": [("S4", "i")],
        #         "S4": [("S3", "i")],
        #     },
        # ),
        # (
        #     """
        #     function adv_vectorization_algo1(A[16,16]) -> [16,16]:
        #         for i in range(0,15,1)
        #             for j in range(0,16,1)
        #                 A[i+1,j] = A[i,j] + 1   # S1
        #             end
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": [("S1", "i")]},
        # ),
        # (
        #     """
        #     function ziv1(A[16], B[16]) -> [16]:
        #         for i in range(0,16,1)
        #             A[0] = 1          # S1
        #             B[i] = A[0] + 2   # S2
        #         end
        #         return B
        #     end
        #     """,
        #     {"S1": [("S1", "ziv"), ("S2", "ziv")], "S2":[("S1", "ziv")]},
        # ),
        # (
        #     """
        #     function weak_siv1(A[16], B[16]) -> [16]:
        #         for i in range(0,16,1)
        #             A[i] = 1          # S1
        #             B[i] = A[5] + 2   # S2
        #         end
        #         return B
        #     end
        #     """,
        #     {"S1": [("S2", "i")], "S2":[("S1", "i")]},
        # ),
        # (
        #     """
        #     function weak_siv2(A[16,16]) -> [16,16]:
        #         for i in range(0,16,1)
        #             for j in range(0,16,1)
        #                 A[i,j] = A[5,j] + 2     # S1
        #             end
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": [("S1", "i")]},
        # ),
        # # my additional tests
        # (
        #     """
        #     function exceed_loop_bounds(A[16]) -> [16]:
        #         for i in range(0,5,1)
        #             A[i+5] = A[i] + 12  # S1
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": []},
        # ),
        # (
        #     """
        #     function unnormalized(A[16]) -> [16]:
        #         for i in range(1, 16, 1)
        #             A[i] = A[i-1] + 12  # S1
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": [("S1", "i")]},
        # ),
        # (
        #     """
        #     function negative(A[16]) -> [16]:
        #         for i in range(0, 16, 1)
        #             A[15-i] = A[i] + 12  # S1
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": []},
        # ),
        # (
        #     """
        #     function negative2(A[17]) -> [17]:
        #         for i in range(0, 16, 1)
        #             A[16-i] = A[8] + 12  # S1
        #         end
        #         return A
        #     end
        #     """,
        #     {"S1": [("S1", "i")]},
        # ),
        # (
        #     """
        #     function no_loop(A[16], B[16]) -> [16]:
        #         for i in range(0, 15, 1)
        #             A[i] = 12 # S1
        #             B[i] = A[i] # S2
        #         end
        #         return B
        #     end
        #     """,
        #     {"S1": [("S2", "i")], "S2": []},
        # ),
        # (
        #     """
        #     function yes_loop(A[16], B[16]) -> [16]:
        #         for j in range(0, 3, 1)
        #             for i in range(0, 15, 1)
        #                 A[i] = 12 # S1
        #                 B[i] = A[i] # S2
        #             end
        #         end
        #         return B
        #     end
        #     """,
        #     {"S1": [('S1', 'j'), ('S2', 'i')], "S2": [("S1", "j"), ("S2", "j")]},
        # ),
        # (
        #     """
        #     function yes_loop2(A[16], B[16]) -> [16]:
        #         for i in range(0, 15, 1)
        #             for j in range(0, 3, 1)
        #                 A[i] = 12 # S1
        #                 B[i] = A[i] # S2
        #             end
        #         end
        #         return B
        #     end
        #     """,
        #     {"S1": [("S1", "j"), ("S2", "i")], "S2": [("S1", "j"), ("S2", "j")]},
        # ),
        (
            """
            function homework2(A[16, 16, 16]) -> [16, 16, 16]:
                for k in range(0, 16, 1)
                    for j in range(0, 16, 1)
                        for i in range(0, 16, 1)
                            A[i+1, j, k] = A[i, j, 5] + 2
                        end
                    end
                end
            end
            """,
            {"S1": [("S1", "i"), ("S1", "k")]},
        ),
    ],
)
def test_vectorize(input_prgm, expected_dependency_graph):
    parser = SimpleLangParser()
    output_prgm = parser.parse(input_prgm)

    # resetting the count to make verification easy
    DependencyGraphNode.count = 0

    # We assume there is only one loop per test case
    dependency_graph = construct_dependency_graph(
        output_prgm.body.bodies[0], dependency_test
    )

    # Simplify dependency graph for verification
    simplified_dependency_graph: dict[str, list[str]] = {}
    for src, children in dependency_graph.items():
        simplified_dependency_graph[src.unique_id] = sorted([
            (child.target.unique_id, child.loop_lvl.name) for child in children
        ])
    assert simplified_dependency_graph == expected_dependency_graph
