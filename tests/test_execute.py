import pytest
import numpy as np

from autovec.codegen import NumpyBuffer
from autovec.simple_lang.compiler import SimpleLang2CCompiler


@pytest.mark.parametrize(
    "input_prgm, input, expected_out",
    [
        # Case: Vectorizable
        (
            """
            function prgm(A[16]) -> [16]:
                for i in range(0,16,1)
                    A[i] = 0
                end
                return A
            end
            """,
            (np.full(shape=(16,), fill_value=5, dtype=np.float64),),
            np.full(shape=(16,), fill_value=0, dtype=np.float64),
        ),
        (
            """
            function prgm(A[16]) -> [16]:
                for i in range(0,15,1)
                    A[i+1] = A[i] + 12
                end
                return A
            end
            """,
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
            ),
            np.array(
                [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133, 145, 157, 169, 181],
                dtype=np.float64,
            ),
        ),
        (
            """
            function prgm(A[16], F[16]) -> [16]:
                for i in range(0,15,1)
                    A[i+1] = F[i]
                    F[i+1] = A[i]
                end
                return F
            end
            """,
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
            ),
            np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                dtype=np.float64,
            ),
        ),
        (
            """
            function prgm(A[16], B[16]) -> [16]:
                for i in range(1,16,1)
                    A[i] = B[i] + 1
                    B[i-1] = A[i] - 5
                end
                return B
            end
            """,
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
            ),
            np.array(
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16],
                dtype=np.float64,
            ),
        ),
        (
            """
            function prgm(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
                for i in range(0,15,1)
                    D[i] = A[i] + 4
                    A[i+1] = B[i] + 6
                    Y[i] = X[i] + D[i]
                    X[i+1] = Y[i] + 9
                end
                return X
            end
            """,
            (
                np.full(shape=(16,), fill_value=1, dtype=np.float64),
                np.full(shape=(16,), fill_value=2, dtype=np.float64),
                np.full(shape=(16,), fill_value=3, dtype=np.float64),
                np.full(shape=(16,), fill_value=4, dtype=np.float64),
                np.full(shape=(16,), fill_value=5, dtype=np.float64),
            ),
            np.array(
                [
                    4,
                    18,
                    39,
                    60,
                    81,
                    102,
                    123,
                    144,
                    165,
                    186,
                    207,
                    228,
                    249,
                    270,
                    291,
                    312,
                ],
                dtype=np.float64,
            ),
        ),
        (
            """
            function prgm(A[16,16]) -> [16,16]:
                for i in range(0,15,1)
                    for j in range(0,16,1)
                        A[i+1,j] = A[i,j] + 1
                    end
                end
                return A
            end
            """,
            (np.full(shape=(16, 16), fill_value=0, dtype=np.float64),),
            np.array([[i] * 16 for i in range(0, 16)], dtype=np.float64),
        ),
        (
            """
            function prgm(A[16,16]) -> [16,16]:
                for i in range(0,16,1)
                    for j in range(0,16,1)
                        A[i,j] = A[5,j] + 2
                    end
                end
                return A
            end
            """,
            (np.array([[i] * 16 for i in range(0, 16)], dtype=np.float64),),
            np.array([[7 if i <= 5 else 9] * 16 for i in range(0, 16)], dtype=np.float64),
        )
    ],
)
def test_execute(input_prgm, input, expected_out):
    compiler = SimpleLang2CCompiler()
    mod = compiler(input_prgm, 8)

    buf_input = [NumpyBuffer(arr) for arr in input]
    result = mod.prgm(*buf_input)
    assert np.allclose(
        result.arr, expected_out
    ), f"Expected {expected_out}, got {result.arr}"
