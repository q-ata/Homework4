#include <stddef.h>
#include <stdint.h>
struct C_CTuple {
    int32_t element_0;
};
struct CNumpyBuffer_float64_1 {
    void* arr;
    double* data;
    size_t length;
    struct C_CTuple shape;
};
typedef void* (*fptr)( size_t* );
fptr alloc_float64_1;
void set_alloc_float64_1(fptr x) { alloc_float64_1 = x; }
struct CNumpyBuffer_float64_1 prgm(struct CNumpyBuffer_float64_1 A) {
    struct CNumpyBuffer_float64_1 A_3 = *(struct CNumpyBuffer_float64_1*)alloc_float64_1((size_t[1]){16});
    for (int64_t vec_i = 0; vec_i < 16; vec_i+=8) {
        size_t linear_idx = 0;
        linear_idx += 0 + vec_i * 1;
        _mm512_i64scatter_pd(A_3.data + linear_idx, _mm512_set_epi64((7*1), (6*1), (5*1), (4*1), (3*1), (2*1), (1*1), (0*1)), _mm512_set1_pd(0), 8);
    }
    for (int64_t vec_i = 0; vec_i < 16; vec_i+=8) {
        size_t linear_idx_2 = 0;
        linear_idx_2 += 0 + vec_i * 1;
        size_t linear_idx_3 = 0;
        linear_idx_3 += 0 + vec_i * 1;
        _mm512_i64scatter_pd(A.data + linear_idx_2, _mm512_set_epi64((7*1), (6*1), (5*1), (4*1), (3*1), (2*1), (1*1), (0*1)), _mm512_i64gather_pd(_mm512_set_epi64((7*1), (6*1), (5*1), (4*1), (3*1), (2*1), (1*1), (0*1)), A_3.data + linear_idx_3, 8), 8);
    }
    return A;
}