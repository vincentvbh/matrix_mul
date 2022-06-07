
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#include <arm_neon.h>

#include "m1cycles.h"

#include "params.h"
#include "tools.h"

#define ASM

#define TEST

struct dimension{
    int32_t dim_i;
    int32_t dim_k;
    int32_t dim_j;
};
struct dimension matrix_dim;

#ifdef ASM
// mla by default
extern "C" void ikj_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ikj_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);

extern "C" void ikj_matmul_addsrc1_asm(int16_t*, int16_t*, int16_t*, struct dimension*, int16_t*);
extern void ikj_matmul_addsrc1_asm(int16_t*, int16_t*, int16_t*, struct dimension*, int16_t*);

extern "C" void ikj_matmul_subnegaccsrc1_asm(int16_t*, int16_t*, int16_t*, struct dimension*, int16_t*, int16_t*);
extern void ikj_matmul_subnegaccsrc1_asm(int16_t*, int16_t*, int16_t*, struct dimension*, int16_t*, int16_t*);


extern "C" void ijk_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ijk_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
#endif

#define STRASSEN_BUFF ( ( (SQUARE_DIM / 2) * (SQUARE_DIM / 2) ) * 4 )
int16_t large_buff[STRASSEN_BUFF];

void Winograd_pre_square_matmul(
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*,
    size_t,
    int16_t*);

int main(void){

    long long start, end;
    long long ns;

    int nonZeros;

    matrix_dim.dim_i = SQUARE_DIM;
    matrix_dim.dim_k = SQUARE_DIM;
    matrix_dim.dim_j = SQUARE_DIM;
    
    int16_t res[SQUARE_DIM][SQUARE_DIM];
    int16_t A[SQUARE_DIM][SQUARE_DIM];
    int16_t B[SQUARE_DIM][SQUARE_DIM];

    int16_t Strassen_Winograd_pre_res[SQUARE_DIM][SQUARE_DIM];

    int16_t A00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t A01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t A10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t A11[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t B00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t B01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t B10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t B11[SQUARE_DIM / 2][SQUARE_DIM / 2];

    int16_t C00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t C01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t C10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t C11[SQUARE_DIM / 2][SQUARE_DIM / 2];

    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            res[i][j] = 0;
        }
    }

    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t k = 0; k < SQUARE_DIM; k++){
            A[i][k] = i * SQUARE_DIM + k;
        }
    }

    for(size_t k = 0; k < SQUARE_DIM; k++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            B[k][j] = k * SQUARE_DIM + j;
        }
    }

    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){

            C00[i][j] = C01[i][j] = 0;
            C10[i][j] = C11[i][j] = 0;

            A00[i][j] = A[i                   ][j                   ];
            A01[i][j] = A[i                   ][j + (SQUARE_DIM / 2)];
            A10[i][j] = A[i + (SQUARE_DIM / 2)][j                   ];
            A11[i][j] = A[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

            B00[i][j] = B[i                   ][j                   ];
            B01[i][j] = B[i                   ][j + (SQUARE_DIM / 2)];
            B10[i][j] = B[i + (SQUARE_DIM / 2)][j                   ];
            B11[i][j] = B[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

        }
    }

    printf("start\n");

// ================================

    setup_rdtsc();

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ikj_matmul_asm(&res[0][0], &A[0][0], &B[0][0], &matrix_dim);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ikj SIMD asm Dense cycles:\n%lld\n", ns);

// ================================

    start = rdtsc();
    for(size_t iter = 0; iter < 16; iter++){
    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){

            A00[i][j] = A[i                   ][j                   ];
            A01[i][j] = A[i                   ][j + (SQUARE_DIM / 2)];
            A10[i][j] = A[i + (SQUARE_DIM / 2)][j                   ];
            A11[i][j] = A[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

        }
    }
    }
    end = rdtsc();
    ns = (end - start);
    printf("A preprocessing cycles:\n%lld\n", ns);

// ================================

    start = rdtsc();
    for(size_t iter = 0; iter < 16; iter++){
    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){
            Strassen_Winograd_pre_res[i                   ][j                   ] = C00[i][j];
            Strassen_Winograd_pre_res[i                   ][j + (SQUARE_DIM / 2)] = C01[i][j];
            Strassen_Winograd_pre_res[i + (SQUARE_DIM / 2)][j                   ] = C10[i][j];
            Strassen_Winograd_pre_res[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)] = C11[i][j];
        }
    }
    }
    end = rdtsc();
    ns = (end - start);
    printf("postprocessing cycles:\n%lld\n", ns);

// ================================

    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){
            B00[i][j] = rand();
            B01[i][j] = rand();
            B10[i][j] = rand();
            B11[i][j] = rand();
        }
    }

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        Winograd_pre_square_matmul(
            &C00[0][0], &C01[0][0],
            &C10[0][0], &C11[0][0],
            &A00[0][0], &A01[0][0], &A10[0][0], &A11[0][0],
            &B00[0][0], &B01[0][0], &B10[0][0], &B11[0][0],
            &A[0][0], &B[0][0],
            SQUARE_DIM / 2,
            large_buff
            );
    }
    end = rdtsc();
    ns = (end - start);
    printf("Winograd with preprocessing cycles:\n%lld\n", ns);

    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){
            Strassen_Winograd_pre_res[i                   ][j                   ] = C00[i][j];
            Strassen_Winograd_pre_res[i                   ][j + (SQUARE_DIM / 2)] = C01[i][j];
            Strassen_Winograd_pre_res[i + (SQUARE_DIM / 2)][j                   ] = C10[i][j];
            Strassen_Winograd_pre_res[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)] = C11[i][j];
        }
    }

#ifdef TEST
    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            if(res[i][j] != Strassen_Winograd_pre_res[i][j]){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    res[i][j], Strassen_Winograd_pre_res[i][j]);
            }
        }
    }
#endif

// ================================

}

void Winograd_pre_square_matmul(
    int16_t *Strassen_C00, int16_t *Strassen_C01, int16_t *Strassen_C10, int16_t *Strassen_C11,
    int16_t *Strassen_A00, int16_t *Strassen_A01, int16_t *Strassen_A10, int16_t *Strassen_A11,
    int16_t *Strassen_B00, int16_t *Strassen_B01, int16_t *Strassen_B10, int16_t *Strassen_B11,
    int16_t *A, int16_t *B,
    size_t dim,
    int16_t* buff){

    int16_t *T0 = buff;
    int16_t *T1 = T0 + (SQUARE_DIM / 2) * (SQUARE_DIM / 2);
    int16_t *T3 = T1 + (SQUARE_DIM / 2) * (SQUARE_DIM / 2);
    int16_t *M3 = T3 + (SQUARE_DIM / 2) * (SQUARE_DIM / 2);

    struct dimension matrix_dim;
    matrix_dim.dim_i = matrix_dim.dim_j = matrix_dim.dim_k = dim;

    // B01 - B00
    matrix_sub_fromM(T1, Strassen_B01, Strassen_B00, dim, dim,
        B + dim, B, 2 * dim);

    matrix_sub(Strassen_C11, Strassen_C11, Strassen_C01, dim, dim);
    // (A10 + A11) * (B01 - B00)
    ikj_matmul_addsrc1_asm(Strassen_C01, Strassen_A10, T1, &matrix_dim,
        Strassen_A11);
    // (A10 + A11) * (B01 - B00)
    matrix_add(Strassen_C11, Strassen_C11, Strassen_C01, dim, dim);

    // A10 - A00
    matrix_sub(T0, Strassen_A10, Strassen_A00, dim, dim);
    // B01 - B11
    matrix_sub_fromM2(T1, Strassen_B01, Strassen_B11, dim, dim,
        B + dim * 2 * dim + dim, 2 * dim);
    // B00 + B11 - B01
    matrix_sub(T3, Strassen_B00, T1, dim, dim);

    // A00 * B00
    ijk_matmul_asm(M3, Strassen_A00, Strassen_B00, &matrix_dim);
    // A00 * B00
    matrix_add(Strassen_C00, Strassen_C00, M3, dim, dim);

    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01)
    ikj_matmul_addsrc1_asm(M3, Strassen_A11, T3, &matrix_dim,
        T0);
    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01)
    matrix_add(Strassen_C01, Strassen_C01, M3, dim, dim);

    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01) + (A10 - A00) * (B01 - B11)
    // =
    // A00 * (B00 - B00 - B11 + B01 - B01 + B11) +
    // A10 * (B00 + B11 - B01 + B01 - B11) +
    // A11 * (B00 + B11 - B01)
    // =
    // A10 * B00 + A11 * (B00 + B11 - B01)
    ikj_matmul_asm(M3, T0, T1, &matrix_dim);
    // A10 * B00 + A11 * (B00 + B11 - B01)
    // (A10 + A11) * (B01 - B00) + A10 * B00 + A11 * (B00 + B11 - B01)
    matrix_addx2(Strassen_C10, Strassen_C11, Strassen_C10, Strassen_C11, M3, dim, dim);

    // B01 + B10 - B00 - B11
    matrix_sub_acc_fromM1(T1, Strassen_B10, Strassen_B00, dim, dim,
        B + dim * 2 * dim, 2 * dim);
    // A00 * B00 + A01 * B10
    ikj_matmul_asm(Strassen_C00, Strassen_A01, Strassen_B10, &matrix_dim);


    // A10 * B00 + A11 * (B00 + B11 - B01) + A11 * (B01 + B10 - B00 - B11)
    // =
    // A10 * B00 + A11 * B10
    ikj_matmul_asm(Strassen_C10, Strassen_A11, T1, &matrix_dim);

    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01) + (A00 + A01 - A10 - A11) * B11
    // =
    // A00 * (B00 - B00 - B11 + B01 + B11) +
    // A01 * B11 +
    // A10 * (B00 + B11 - B01 - B11) +
    // A11 * (B00 + B11 - B01 - B11)
    // =
    // A00 * B01 + A01 * B11 + A10 * (B00 - B01) + A11 * (B00 - B01)
    ikj_matmul_subnegaccsrc1_asm(Strassen_C01, T0, Strassen_B11, &matrix_dim,
            Strassen_A01, Strassen_A11);

}











