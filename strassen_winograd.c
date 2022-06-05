
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

extern "C" void ijk_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ijk_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);

extern "C" void ijk_matmla_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ijk_matmla_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
#endif

void Strassen_square_matmul(
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    size_t);

void Strassen_Winograd_square_matmul(
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    size_t);

void Strassen_Winograd_pre_square_matmul(
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*, int16_t*, int16_t*,
    int16_t*, int16_t*,
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

    int16_t Strassen_Winograd_res[SQUARE_DIM][SQUARE_DIM];

    int16_t Strassen_A00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_A01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_A10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_A11[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_B11[SQUARE_DIM / 2][SQUARE_DIM / 2];

    int16_t Strassen_Winograd_C00[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_Winograd_C01[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_Winograd_C10[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t Strassen_Winograd_C11[SQUARE_DIM / 2][SQUARE_DIM / 2];


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

            Strassen_Winograd_C00[i][j] = Strassen_Winograd_C01[i][j] = 0;
            Strassen_Winograd_C10[i][j] = Strassen_Winograd_C11[i][j] = 0;

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
            Strassen_Winograd_res[i                   ][j                   ] = Strassen_Winograd_C00[i][j];
            Strassen_Winograd_res[i                   ][j + (SQUARE_DIM / 2)] = Strassen_Winograd_C01[i][j];
            Strassen_Winograd_res[i + (SQUARE_DIM / 2)][j                   ] = Strassen_Winograd_C10[i][j];
            Strassen_Winograd_res[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)] = Strassen_Winograd_C11[i][j];
        }
    }
    }
    end = rdtsc();
    ns = (end - start);
    printf("postprocessing cycles:\n%lld\n", ns);

// ================================

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        Strassen_Winograd_square_matmul(
            &Strassen_Winograd_C00[0][0], &Strassen_Winograd_C01[0][0], &Strassen_Winograd_C10[0][0], &Strassen_Winograd_C11[0][0],
            &Strassen_A00[0][0], &Strassen_A01[0][0], &Strassen_A10[0][0], &Strassen_A11[0][0],
            &Strassen_B00[0][0], &Strassen_B01[0][0], &Strassen_B10[0][0], &Strassen_B11[0][0],
            SQUARE_DIM / 2
            );
    }
    end = rdtsc();
    ns = (end - start);
    printf("Strassen_Winograd cycles:\n%lld\n", ns);

    for(size_t i = 0; i < SQUARE_DIM / 2; i++){
        for(size_t j = 0; j < SQUARE_DIM / 2; j++){
            Strassen_Winograd_res[i                   ][j                   ] = Strassen_Winograd_C00[i][j];
            Strassen_Winograd_res[i                   ][j + (SQUARE_DIM / 2)] = Strassen_Winograd_C01[i][j];
            Strassen_Winograd_res[i + (SQUARE_DIM / 2)][j                   ] = Strassen_Winograd_C10[i][j];
            Strassen_Winograd_res[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)] = Strassen_Winograd_C11[i][j];
        }
    }

#ifdef TEST
    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            if(A[i][j] != Strassen_Winograd_res[i][j]){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    A[i][j], Strassen_Winograd_res[i][j]);
            }
        }
    }
#endif

// ================================

}

void Strassen_Winograd_square_matmul(
    int16_t *Strassen_C00, int16_t *Strassen_C01, int16_t *Strassen_C10, int16_t *Strassen_C11,
    int16_t *Strassen_A00, int16_t *Strassen_A01, int16_t *Strassen_A10, int16_t *Strassen_A11,
    int16_t *Strassen_B00, int16_t *Strassen_B01, int16_t *Strassen_B10, int16_t *Strassen_B11,
    size_t dim){

    int16_t T0[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T1[SQUARE_DIM / 2][SQUARE_DIM / 2];
    // int16_t T2[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T3[SQUARE_DIM / 2][SQUARE_DIM / 2];

    int16_t M3[SQUARE_DIM / 2][SQUARE_DIM / 2];

    struct dimension matrix_dim;
    matrix_dim.dim_i = matrix_dim.dim_j = matrix_dim.dim_k = dim;

    // matrix_add(&T0[0][0], Strassen_A10, Strassen_A11, dim, dim);
    matrix_sub(&T1[0][0], Strassen_B01, Strassen_B00, dim, dim);

    matrix_sub(Strassen_C11, Strassen_C11, Strassen_C01, dim, dim);

    // A00 * B01 + A01 * B11 + A10 * (B00 - B01) + A11 * (B00 - B01) +
    // (A10 + A11) * (B01 - B00)
    // =
    // A00 * B01 + A01 * B11
    ikj_matmul_addsrc1_asm(Strassen_C01, Strassen_A10, &T1[0][0], &matrix_dim, Strassen_A11);
    // ikj_matmul_asm(Strassen_C01, &T0[0][0], &T1[0][0], &matrix_dim);

    // A10 * B00 + A11 * (B00 + B11 - B01) + (A10 + A11) * (B01 - B00)
    // =
    // A10 * B01 + A11 * B11
    // matrix_add(Strassen_C11, Strassen_C11, Strassen_C01, dim, dim);

    // A10 - A00
    matrix_sub(&T0[0][0], Strassen_A10, Strassen_A00, dim, dim);


    // B01 - B11
    matrix_sub(&T1[0][0], Strassen_B01, Strassen_B11, dim, dim);


    // A00 * B00
    ijk_matmul_asm(&M3[0][0], Strassen_A00, Strassen_B00, &matrix_dim);
    // A00 * B00
    matrix_add(Strassen_C00, Strassen_C00, &M3[0][0], dim, dim);

    // A10 + A11 - A00
    // matrix_add(&T2[0][0], Strassen_A11, &T0[0][0], dim, dim);
    // B00 + B11 - B01
    matrix_sub(&T3[0][0], Strassen_B00, &T1[0][0], dim, dim);

    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01)
    ikj_matmul_addsrc1_asm(&M3[0][0], Strassen_A11, &T3[0][0], &matrix_dim, &T0[0][0]);
    // ikj_matmul_asm(&M3[0][0], &T2[0][0], &T3[0][0], &matrix_dim);
    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01)

    // A10 * B00 + A11 * (B00 + B11 - B01) + (A10 + A11) * (B01 - B00)
    // =
    // A10 * B01 + A11 * B11
    // matrix_add(Strassen_C11, Strassen_C11, Strassen_C01, dim, dim);
    // matrix_add(Strassen_C01, &M3[0][0], Strassen_C01, dim, dim);
    matrix_addx2(Strassen_C11, Strassen_C01, Strassen_C11, &M3[0][0], Strassen_C01, dim, dim);

    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01) + (A10 - A00) * (B01 - B11)
    // =
    // A00 * (B00 - B00 - B11 + B01 - B01 + B11) +
    // A10 * (B00 + B11 - B01 + B01 - B11) +
    // A11 * (B00 + B11 - B01)
    // =
    // A10 * B00 + A11 * (B00 + B11 - B01)
    ikj_matmul_asm(&M3[0][0], &T0[0][0], &T1[0][0], &matrix_dim);
    // A10 * B00 + A11 * (B00 + B11 - B01)
    // A10 * B00 + A11 * (B00 + B11 - B01)
    matrix_addx2(Strassen_C10, Strassen_C11, Strassen_C10, Strassen_C11, &M3[0][0], dim, dim);



    // B01 + B10 - B00 - B11
    matrix_sub_acc(&T1[0][0], Strassen_B10, Strassen_B00, dim, dim);
    // A00 * B00 + A01 * B10
    ikj_matmul_asm(Strassen_C00, Strassen_A01, Strassen_B10, &matrix_dim);


    // A10 * B00 + A11 * (B00 + B11 - B01) + A11 * (B01 + B10 - B00 - B11)
    // =
    // A10 * B00 + A11 * B10
    ikj_matmul_asm(Strassen_C10, Strassen_A11, &T1[0][0], &matrix_dim);

    // A00 + A01 - A10 - A11
    matrix_sub_negacc(&T0[0][0], Strassen_A01, Strassen_A11, dim, dim);
    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01) + (A00 + A01 - A10 - A11) * B11
    // =
    // A00 * (B00 - B00 - B11 + B01 + B11) +
    // A01 * B11 +
    // A10 * (B00 + B11 - B01 - B11) +
    // A11 * (B00 + B11 - B01 - B11)
    // =
    // A00 * B01 + A01 * B11 + A10 * (B00 - B01) + A11 * (B00 - B01)
    ikj_matmul_asm(Strassen_C01, &T0[0][0], Strassen_B11, &matrix_dim);

}












