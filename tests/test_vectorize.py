import pytest

from autovec.simple_lang.vectorizer.vectorize import vectorize
from autovec.simple_lang.vectorizer.dependency_testing import dependency_test
from autovec.simple_lang.parser import SimpleLangParser


@pytest.mark.parametrize(
    "input_prgm, expected_output_prgm",
    [
        # Case: Vectorizable
        (
            """
            function vectorizable1(A[16]) -> [16]:
                for i in range(0,16,1)
                    A[i] = 0
                end
                return A
            end
            """,
            """
            function vectorizable1(A[16]) -> [16]:
                A[0:16] = 0
                return A
            end
            """,
        ),
        # Case: Unvectorizable
        (
            """
            function unvectorizable1(A[16]) -> [16]:
                for i in range(0,15,1)
                    A[i+1] = A[i] + 12
                end
                return A
            end
            """,
            """
            function unvectorizable1(A[16]) -> [16]:
                for i in range(0,15,1)
                    A[i+1] = A[i] + 12
                end
                return A
            end
            """,
        ),
        (
            """
            function unvectorizable2(A[16], F[16]) -> [16]:
                for i in range(0,15,1)
                    A[i+1] = F[i]
                    F[i+1] = A[i]
                end
                return F
            end
            """,
            """
            function unvectorizable2(A[16], F[16]) -> [16]:
                for i in range(0,15,1)
                    A[i+1] = F[i]
                    F[i+1] = A[i]
                end
                return F
            end
            """,
        ),

        # Case: Vectorization through loop splitting
        (
            """
            function loop_splitting1(A[16], B[16]) -> [16]:
                for i in range(1,16,1)
                    A[i] = B[i] + 1
                    B[i-1] = A[i] - 5
                end
                return B
            end
            """,
            """
            function loop_splitting1(A[16], B[16]) -> [16]:
                A[1:16] = B[1:16] + 1
                B[0:15] = A[1:16] - 5
                return B
            end
            """,
        ),
        # Case: Simple Vectorization algorithm
        (
            """
            function simple_vectorization_algo1(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
                for i in range(0,15,1)
                    D[i] = A[i] + 4
                    A[i+1] = B[i] + 6
                    Y[i] = X[i] + D[i]
                    X[i+1] = Y[i] + 9
                end
                return X
            end
            """,
            """
            function simple_vectorization_algo1(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
                A[1:16] = B[0:15] + 6
                D[0:15] = A[0:15] + 4
                for i in range(0,15,1)
                    Y[i] = X[i] + D[i]
                    X[i+1] = Y[i] + 9
                end
                return X
            end
            """,
        ),
        # Case: Advanced Vectorization algorithm
        (
            """
            function adv_vectorization_algo1(A[16,16]) -> [16,16]:
                for i in range(0,15,1)
                    for j in range(0,16,1)
                        A[i+1,j] = A[i,j] + 1
                    end
                end
                return A
            end
            """,
            """
            function adv_vectorization_algo1(A[16,16]) -> [16,16]:
                for i in range(0,15,1)
                    A[i+1,0:16] = A[i,0:16] + 1
                end
                return A
            end
            """,
        ),
    ],
)
def test_vectorize(input_prgm, expected_output_prgm):
    parsed_prgm = SimpleLangParser().parse(input_prgm)
    vectorized_prgm = vectorize(parsed_prgm, dependency_test)
    assert vectorized_prgm == SimpleLangParser().parse(expected_output_prgm)
