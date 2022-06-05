
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
extern "C" void ikj_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ikj_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);

extern "C" void ijk_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ijk_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
#endif

void Strassen_square_matmul(
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    size_t);

int main(void){

    long long start, end;
    long long ns;

    int nonZeros;

    matrix_dim.dim_i = SQUARE_DIM;
    matrix_dim.dim_k = SQUARE_DIM;
    matrix_dim.dim_j = SQUARE_DIM;
    
    int16_t A[SQUARE_DIM][SQUARE_DIM];
    int16_t A0[SQUARE_DIM][SQUARE_DIM];
    int16_t A1[SQUARE_DIM][SQUARE_DIM];

    int16_t Strassen_res[SQUARE_DIM][SQUARE_DIM];

    int16_t Strassen_A00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_A01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_A10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_A11[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B11[SQUARE_DIM / 2][SQUARE_DIM / 2];

    int16_t Strassen_C00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_C01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_C10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_C11[SQUARE_DIM / 2][SQUARE_DIM / 2];

    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            A[i][j] = 0;
        }
    }

    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t k = 0; k < SQUARE_DIM; k++){
            A0[i][k] = i * SQUARE_DIM + k;
        }
    }

    for(size_t k = 0; k < SQUARE_DIM; k++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            A1[k][j] = k * SQUARE_DIM + j;
        }
    }

    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){

            Strassen_C00[i][j] = Strassen_C01[i][j] = 0;
            Strassen_C10[i][j] = Strassen_C11[i][j] = 0;

            Strassen_A00[i][j] = A0[i                   ][j                   ];
            Strassen_A01[i][j] = A0[i                   ][j + (SQUARE_DIM / 2)];
            Strassen_A10[i][j] = A0[i + (SQUARE_DIM / 2)][j                   ];
            Strassen_A11[i][j] = A0[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

            Strassen_B00[i][j] = A1[i                   ][j                   ];
            Strassen_B01[i][j] = A1[i                   ][j + (SQUARE_DIM / 2)];
            Strassen_B10[i][j] = A1[i + (SQUARE_DIM / 2)][j                   ];
            Strassen_B11[i][j] = A1[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

        }
    }

    printf("start\n");

// ================================

    setup_rdtsc();

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ikj_matmul_asm(&A[0][0], &A0[0][0], &A1[0][0], &matrix_dim);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ikj SIMD asm Dense cycles:\n%lld\n", ns);

// ================================

    start = rdtsc();
    for(size_t iter = 0; iter < 16; iter++){
    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){

            Strassen_A00[i][j] = A0[i                   ][j                   ];
            Strassen_A01[i][j] = A0[i                   ][j + (SQUARE_DIM / 2)];
            Strassen_A10[i][j] = A0[i + (SQUARE_DIM / 2)][j                   ];
            Strassen_A11[i][j] = A0[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

            Strassen_B00[i][j] = A1[i                   ][j                   ];
            Strassen_B01[i][j] = A1[i                   ][j + (SQUARE_DIM / 2)];
            Strassen_B10[i][j] = A1[i + (SQUARE_DIM / 2)][j                   ];
            Strassen_B11[i][j] = A1[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

        }
    }
    }
    end = rdtsc();
    ns = (end - start);
    printf("preprocessing cycles:\n%lld\n", ns);

// ================================

    start = rdtsc();
    for(size_t iter = 0; iter < 16; iter++){
    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){
            Strassen_res[i                   ][j                   ] = Strassen_C00[i][j];
            Strassen_res[i                   ][j + (SQUARE_DIM / 2)] = Strassen_C01[i][j];
            Strassen_res[i + (SQUARE_DIM / 2)][j                   ] = Strassen_C10[i][j];
            Strassen_res[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)] = Strassen_C11[i][j];
        }
    }
    }
    end = rdtsc();
    ns = (end - start);
    printf("postprocessing cycles:\n%lld\n", ns);

// ================================

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        Strassen_square_matmul(
            &Strassen_C00[0][0], &Strassen_C01[0][0], &Strassen_C10[0][0], &Strassen_C11[0][0],
            &Strassen_A00[0][0], &Strassen_A01[0][0], &Strassen_A10[0][0], &Strassen_A11[0][0],
            &Strassen_B00[0][0], &Strassen_B01[0][0], &Strassen_B10[0][0], &Strassen_B11[0][0],
            SQUARE_DIM / 2
            );
    }
    end = rdtsc();
    ns = (end - start);
    printf("Strassen cycles:\n%lld\n", ns);

    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){
            Strassen_res[i                   ][j                   ] = Strassen_C00[i][j];
            Strassen_res[i                   ][j + (SQUARE_DIM / 2)] = Strassen_C01[i][j];
            Strassen_res[i + (SQUARE_DIM / 2)][j                   ] = Strassen_C10[i][j];
            Strassen_res[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)] = Strassen_C11[i][j];
        }
    }

#ifdef TEST
    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            if(A[i][j] != Strassen_res[i][j]){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    A[i][j], Strassen_res[i][j]);
            }
        }
    }
#endif

// ================================

}

void Strassen_square_matmul(
    int16_t *Strassen_C00, int16_t *Strassen_C01, int16_t *Strassen_C10, int16_t *Strassen_C11,
    int16_t *Strassen_A00, int16_t *Strassen_A01, int16_t *Strassen_A10, int16_t *Strassen_A11,
    int16_t *Strassen_B00, int16_t *Strassen_B01, int16_t *Strassen_B10, int16_t *Strassen_B11,
    size_t dim){

    int16_t T0[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T1[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T2[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T3[SQUARE_DIM / 2][SQUARE_DIM / 2];


    int16_t M0[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t M1[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t M2[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t M3[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t M4[SQUARE_DIM / 2][SQUARE_DIM / 2];

    for(size_t i = 0; i < dim; i++){
        for(size_t j = 0; j < dim; j++){
            M0[i][j] = M1[i][j] = M2[i][j] = M3[i][j] = M4[i][j] = 0;
        }
    }

    struct dimension matrix_dim;
    matrix_dim.dim_i = matrix_dim.dim_j = matrix_dim.dim_k = dim;

    matrix_add(&T0[0][0], Strassen_A00, Strassen_A11, dim, dim);
    matrix_add(&T1[0][0], Strassen_B00, Strassen_B11, dim, dim);
    ikj_matmul_asm(&M0[0][0], &T0[0][0], &T1[0][0], &matrix_dim);

    matrix_add(&T2[0][0], Strassen_A10, Strassen_A11, dim, dim);
    ikj_matmul_asm(&M1[0][0], &T2[0][0], Strassen_B00, &matrix_dim);

    matrix_sub(&T3[0][0], Strassen_B01, Strassen_B11, dim, dim);
    ikj_matmul_asm(&M2[0][0], Strassen_A00, &T3[0][0], &matrix_dim);

    matrix_sub(&T0[0][0], Strassen_B10, Strassen_B00, dim, dim);
    ikj_matmul_asm(&M3[0][0], Strassen_A11, &T0[0][0], &matrix_dim);

    matrix_add(&T1[0][0], Strassen_A00, Strassen_A01, dim, dim);
    ikj_matmul_asm(&M4[0][0], &T1[0][0], Strassen_B11, &matrix_dim);

    matrix_add_acc(Strassen_C01, &M2[0][0], &M4[0][0], dim, dim);

    matrix_add_acc(Strassen_C10, &M1[0][0], &M3[0][0], dim, dim);

    matrix_sub(&T0[0][0], Strassen_A01, Strassen_A11, dim, dim);
    matrix_add(&T1[0][0], Strassen_B10, Strassen_B11, dim, dim);
    matrix_add_acc(Strassen_C00, &M0[0][0], &M3[0][0], dim, dim);
    matrix_sub(Strassen_C00, Strassen_C00, &M4[0][0], dim, dim);
    ikj_matmul_asm(Strassen_C00, &T0[0][0], &T1[0][0], &matrix_dim);

    matrix_sub(&T2[0][0], Strassen_A10, Strassen_A00, dim, dim);
    matrix_add(&T3[0][0], Strassen_B00, Strassen_B01, dim, dim);
    matrix_add_acc(Strassen_C11, &M0[0][0], &M2[0][0], dim, dim);
    matrix_sub(Strassen_C11, Strassen_C11, &M1[0][0], dim, dim);
    ikj_matmul_asm(Strassen_C11, &T2[0][0], &T3[0][0], &matrix_dim);


}













