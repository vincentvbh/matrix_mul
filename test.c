
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include <arm_neon.h>

#include "m1cycles.h"

#include "params.h"

#define BLOCK8
#define ASM

#define TEST

void matrix_add(int16_t *des, int16_t *src1, int16_t *src2, size_t, size_t);
void matrix_sub(int16_t *des, int16_t *src1, int16_t *src2, size_t, size_t);

void matrix_add_acc(int16_t *des, int16_t *src1, int16_t *src2, size_t, size_t);
void matrix_sub_acc(int16_t *des, int16_t *src1, int16_t *src2, size_t, size_t);

void matrix_sub_negacc(int16_t *des, int16_t *src1, int16_t *src2, size_t, size_t);


struct dimension{
    int32_t dim_i;
    int32_t dim_k;
    int32_t dim_j;
};
struct dimension matrix_dim;

#ifdef ASM
extern "C" void ikj_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ikj_matmul_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
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
    int16_t Strassen_Winograd_res[SQUARE_DIM][SQUARE_DIM];

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
            Strassen_C00[i][j] = Strassen_C01[i][j] = Strassen_C10[i][j] = Strassen_C11[i][j] = 0;
            Strassen_Winograd_C00[i][j] = Strassen_Winograd_C01[i][j] = Strassen_Winograd_C10[i][j] = Strassen_Winograd_C11[i][j] = 0;


            Strassen_A00[i][j] = A0[i              ][j              ];
            Strassen_A01[i][j] = A0[i              ][j + (SQUARE_DIM / 2)];
            Strassen_A10[i][j] = A0[i + (SQUARE_DIM / 2)][j              ];
            Strassen_A11[i][j] = A0[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

            Strassen_B00[i][j] = A1[i              ][j              ];
            Strassen_B01[i][j] = A1[i              ][j + (SQUARE_DIM / 2)];
            Strassen_B10[i][j] = A1[i + (SQUARE_DIM / 2)][j              ];
            Strassen_B11[i][j] = A1[i + (SQUARE_DIM / 2)][j + (SQUARE_DIM / 2)];

        }
    }

    printf("start\n");

    setup_rdtsc();

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ikj_matmul_asm(&A[0][0], &A0[0][0], &A1[0][0], &matrix_dim);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ikj SIMD asm Dense cycles:\n%lld\n", ns);


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
            Strassen_res[i              ][j              ] = Strassen_C00[i][j];
            Strassen_res[i              ][j + (SQUARE_DIM / 2)] = Strassen_C01[i][j];
            Strassen_res[i + (SQUARE_DIM / 2)][j              ] = Strassen_C10[i][j];
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
            Strassen_Winograd_res[i              ][j              ] = Strassen_Winograd_C00[i][j];
            Strassen_Winograd_res[i              ][j + (SQUARE_DIM / 2)] = Strassen_Winograd_C01[i][j];
            Strassen_Winograd_res[i + (SQUARE_DIM / 2)][j              ] = Strassen_Winograd_C10[i][j];
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

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        matrix_add(&A[0][0], &A[0][0], &A[0][0], SQUARE_DIM / 2, SQUARE_DIM / 2);
    }
    end = rdtsc();
    ns = (end - start);
    printf("add matrices cycles:\n%lld\n", ns);

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        matrix_sub(&A[0][0], &A[0][0], &A[0][0], SQUARE_DIM / 2, SQUARE_DIM / 2);
    }
    end = rdtsc();
    ns = (end - start);
    printf("sub matrices cycles:\n%lld\n", ns);

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        matrix_add_acc(&A[0][0], &A[0][0], &A[0][0], SQUARE_DIM / 2, SQUARE_DIM / 2);
    }
    end = rdtsc();
    ns = (end - start);
    printf("add acc matrices cycles:\n%lld\n", ns);

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        matrix_sub_acc(&A[0][0], &A[0][0], &A[0][0], SQUARE_DIM / 2, SQUARE_DIM / 2);
    }
    end = rdtsc();
    ns = (end - start);
    printf("sub acc matrices cycles:\n%lld\n", ns);

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        matrix_sub_negacc(&A[0][0], &A[0][0], &A[0][0], SQUARE_DIM / 2, SQUARE_DIM / 2);
    }
    end = rdtsc();
    ns = (end - start);
    printf("sub negacc matrices cycles:\n%lld\n", ns);

}



void matrix_add(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;
    int16x8_t a3;
    int16x8_t a4;
    int16x8_t a5;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;
    int16x8_t b3;
    int16x8_t b4;
    int16x8_t b5;


    for(size_t i = 0; i < (dim_i * dim_j); i += 48){

        a0 = vld1q_s16(src1 + i + 0 * 8);
        a1 = vld1q_s16(src1 + i + 1 * 8);
        a2 = vld1q_s16(src1 + i + 2 * 8);
        a3 = vld1q_s16(src1 + i + 3 * 8);
        a4 = vld1q_s16(src1 + i + 4 * 8);
        a5 = vld1q_s16(src1 + i + 5 * 8);

        b0 = vld1q_s16(src2 + i + 0 * 8);
        b1 = vld1q_s16(src2 + i + 1 * 8);
        b2 = vld1q_s16(src2 + i + 2 * 8);
        b3 = vld1q_s16(src2 + i + 3 * 8);
        b4 = vld1q_s16(src2 + i + 4 * 8);
        b5 = vld1q_s16(src2 + i + 5 * 8);

        vst1q_s16(des + i + 0 * 8, a0 + b0);
        vst1q_s16(des + i + 1 * 8, a1 + b1);
        vst1q_s16(des + i + 2 * 8, a2 + b2);
        vst1q_s16(des + i + 3 * 8, a3 + b3);
        vst1q_s16(des + i + 4 * 8, a4 + b4);
        vst1q_s16(des + i + 5 * 8, a5 + b5);

    }

}

void matrix_sub(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;
    int16x8_t a3;
    int16x8_t a4;
    int16x8_t a5;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;
    int16x8_t b3;
    int16x8_t b4;
    int16x8_t b5;


    for(size_t i = 0; i < (dim_i * dim_j); i += 48){

        a0 = vld1q_s16(src1 + i + 0 * 8);
        a1 = vld1q_s16(src1 + i + 1 * 8);
        a2 = vld1q_s16(src1 + i + 2 * 8);
        a3 = vld1q_s16(src1 + i + 3 * 8);
        a4 = vld1q_s16(src1 + i + 4 * 8);
        a5 = vld1q_s16(src1 + i + 5 * 8);

        b0 = vld1q_s16(src2 + i + 0 * 8);
        b1 = vld1q_s16(src2 + i + 1 * 8);
        b2 = vld1q_s16(src2 + i + 2 * 8);
        b3 = vld1q_s16(src2 + i + 3 * 8);
        b4 = vld1q_s16(src2 + i + 4 * 8);
        b5 = vld1q_s16(src2 + i + 5 * 8);

        vst1q_s16(des + i + 0 * 8, a0 - b0);
        vst1q_s16(des + i + 1 * 8, a1 - b1);
        vst1q_s16(des + i + 2 * 8, a2 - b2);
        vst1q_s16(des + i + 3 * 8, a3 - b3);
        vst1q_s16(des + i + 4 * 8, a4 - b4);
        vst1q_s16(des + i + 5 * 8, a5 - b5);

    }

}

void matrix_add_acc(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;
    int16x8_t a3;
    int16x8_t a4;
    int16x8_t a5;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;
    int16x8_t b3;
    int16x8_t b4;
    int16x8_t b5;

    int16x8_t c0;
    int16x8_t c1;
    int16x8_t c2;
    int16x8_t c3;
    int16x8_t c4;
    int16x8_t c5;


    for(size_t i = 0; i < (dim_i * dim_j); i += 48){

        c0 = vld1q_s16(des + i + 0 * 8);
        c1 = vld1q_s16(des + i + 1 * 8);
        c2 = vld1q_s16(des + i + 2 * 8);
        c3 = vld1q_s16(des + i + 3 * 8);
        c4 = vld1q_s16(des + i + 4 * 8);
        c5 = vld1q_s16(des + i + 5 * 8);

        a0 = vld1q_s16(src1 + i + 0 * 8);
        a1 = vld1q_s16(src1 + i + 1 * 8);
        a2 = vld1q_s16(src1 + i + 2 * 8);
        a3 = vld1q_s16(src1 + i + 3 * 8);
        a4 = vld1q_s16(src1 + i + 4 * 8);
        a5 = vld1q_s16(src1 + i + 5 * 8);

        b0 = vld1q_s16(src2 + i + 0 * 8);
        b1 = vld1q_s16(src2 + i + 1 * 8);
        b2 = vld1q_s16(src2 + i + 2 * 8);
        b3 = vld1q_s16(src2 + i + 3 * 8);
        b4 = vld1q_s16(src2 + i + 4 * 8);
        b5 = vld1q_s16(src2 + i + 5 * 8);

        vst1q_s16(des + i + 0 * 8, c0 + a0 + b0);
        vst1q_s16(des + i + 1 * 8, c1 + a1 + b1);
        vst1q_s16(des + i + 2 * 8, c2 + a2 + b2);
        vst1q_s16(des + i + 3 * 8, c3 + a3 + b3);
        vst1q_s16(des + i + 4 * 8, c4 + a4 + b4);
        vst1q_s16(des + i + 5 * 8, c5 + a5 + b5);

    }

}

void matrix_sub_acc(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;
    int16x8_t a3;
    int16x8_t a4;
    int16x8_t a5;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;
    int16x8_t b3;
    int16x8_t b4;
    int16x8_t b5;

    int16x8_t c0;
    int16x8_t c1;
    int16x8_t c2;
    int16x8_t c3;
    int16x8_t c4;
    int16x8_t c5;


    for(size_t i = 0; i < (dim_i * dim_j); i += 48){

        c0 = vld1q_s16(des + i + 0 * 8);
        c1 = vld1q_s16(des + i + 1 * 8);
        c2 = vld1q_s16(des + i + 2 * 8);
        c3 = vld1q_s16(des + i + 3 * 8);
        c4 = vld1q_s16(des + i + 4 * 8);
        c5 = vld1q_s16(des + i + 5 * 8);

        a0 = vld1q_s16(src1 + i + 0 * 8);
        a1 = vld1q_s16(src1 + i + 1 * 8);
        a2 = vld1q_s16(src1 + i + 2 * 8);
        a3 = vld1q_s16(src1 + i + 3 * 8);
        a4 = vld1q_s16(src1 + i + 4 * 8);
        a5 = vld1q_s16(src1 + i + 5 * 8);

        b0 = vld1q_s16(src2 + i + 0 * 8);
        b1 = vld1q_s16(src2 + i + 1 * 8);
        b2 = vld1q_s16(src2 + i + 2 * 8);
        b3 = vld1q_s16(src2 + i + 3 * 8);
        b4 = vld1q_s16(src2 + i + 4 * 8);
        b5 = vld1q_s16(src2 + i + 5 * 8);

        vst1q_s16(des + i + 0 * 8, c0 + a0 - b0);
        vst1q_s16(des + i + 1 * 8, c1 + a1 - b1);
        vst1q_s16(des + i + 2 * 8, c2 + a2 - b2);
        vst1q_s16(des + i + 3 * 8, c3 + a3 - b3);
        vst1q_s16(des + i + 4 * 8, c4 + a4 - b4);
        vst1q_s16(des + i + 5 * 8, c5 + a5 - b5);

    }

}

void matrix_sub_negacc(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;
    int16x8_t a3;
    int16x8_t a4;
    int16x8_t a5;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;
    int16x8_t b3;
    int16x8_t b4;
    int16x8_t b5;

    int16x8_t c0;
    int16x8_t c1;
    int16x8_t c2;
    int16x8_t c3;
    int16x8_t c4;
    int16x8_t c5;


    for(size_t i = 0; i < (dim_i * dim_j); i += 48){

        c0 = vld1q_s16(des + i + 0 * 8);
        c1 = vld1q_s16(des + i + 1 * 8);
        c2 = vld1q_s16(des + i + 2 * 8);
        c3 = vld1q_s16(des + i + 3 * 8);
        c4 = vld1q_s16(des + i + 4 * 8);
        c5 = vld1q_s16(des + i + 5 * 8);

        a0 = vld1q_s16(src1 + i + 0 * 8);
        a1 = vld1q_s16(src1 + i + 1 * 8);
        a2 = vld1q_s16(src1 + i + 2 * 8);
        a3 = vld1q_s16(src1 + i + 3 * 8);
        a4 = vld1q_s16(src1 + i + 4 * 8);
        a5 = vld1q_s16(src1 + i + 5 * 8);

        b0 = vld1q_s16(src2 + i + 0 * 8);
        b1 = vld1q_s16(src2 + i + 1 * 8);
        b2 = vld1q_s16(src2 + i + 2 * 8);
        b3 = vld1q_s16(src2 + i + 3 * 8);
        b4 = vld1q_s16(src2 + i + 4 * 8);
        b5 = vld1q_s16(src2 + i + 5 * 8);

        vst1q_s16(des + i + 0 * 8, a0 - b0 - c0);
        vst1q_s16(des + i + 1 * 8, a1 - b1 - c1);
        vst1q_s16(des + i + 2 * 8, a2 - b2 - c2);
        vst1q_s16(des + i + 3 * 8, a3 - b3 - c3);
        vst1q_s16(des + i + 4 * 8, a4 - b4 - c4);
        vst1q_s16(des + i + 5 * 8, a5 - b5 - c5);

    }

}

void matrix_add_sub_acc(int16_t *des, int16_t *src1, int16_t *src2, int16_t *src3,
    size_t dim_i, size_t dim_j){

    for(size_t i = 0; i < dim_i; i++){
        for(size_t j = 0; j < dim_j; j += 8){
            vst1q_s16( des + i * dim_j + j,
                vld1q_s16(des + i * dim_j + j) +
                vld1q_s16(src1 + i * dim_j + j) + vld1q_s16(src2 + i * dim_j + j) - vld1q_s16(src3 + i * dim_j + j)
            );
        }
    }

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

// matrix_add: 4
// matrix_sub: 4
// matrix_add_acc: 2
// matrix_sub_acc: 1
// matrix_sub_negacc: 1
void Strassen_Winograd_square_matmul(
    int16_t *Strassen_C00, int16_t *Strassen_C01, int16_t *Strassen_C10, int16_t *Strassen_C11,
    int16_t *Strassen_A00, int16_t *Strassen_A01, int16_t *Strassen_A10, int16_t *Strassen_A11,
    int16_t *Strassen_B00, int16_t *Strassen_B01, int16_t *Strassen_B10, int16_t *Strassen_B11,
    size_t dim){

    int16_t T0[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T1[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T2[SQUARE_DIM / 2][SQUARE_DIM / 2];
    int16_t T3[SQUARE_DIM / 2][SQUARE_DIM / 2];

    int16_t M3[SQUARE_DIM / 2][SQUARE_DIM / 2];

    for(size_t i = 0; i < dim; i++){
        for(size_t j = 0; j < dim; j++){
            M3[i][j] = 0;
        }
    }

    struct dimension matrix_dim;
    matrix_dim.dim_i = matrix_dim.dim_j = matrix_dim.dim_k = dim;


    // A00 * B00
    ikj_matmul_asm(&M3[0][0], Strassen_A00, Strassen_B00, &matrix_dim);
    matrix_add(Strassen_C00, Strassen_C00, &M3[0][0], dim, dim);

    // A10 - A00
    matrix_sub(&T0[0][0], Strassen_A10, Strassen_A00, dim, dim);
    // B01 - B11
    matrix_sub(&T1[0][0], Strassen_B01, Strassen_B11, dim, dim);

    // A10 + A11 - A00
    matrix_add(&T2[0][0], &T0[0][0], Strassen_A11, dim, dim);
    // B00 + B11 - B01
    matrix_sub(&T3[0][0], Strassen_B00, &T1[0][0], dim, dim);
    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01)
    ikj_matmul_asm(&M3[0][0], &T2[0][0], &T3[0][0], &matrix_dim);


    matrix_add(Strassen_C01, Strassen_C01, &M3[0][0], dim, dim);
    // A00 * B00 + (A10 + A11 - A00) * (B00 + B11 - B01) + (A10 - A00) * (B01 - B11)
    // =
    // A00 * (B00 - B00 - B11 + B01 - B01 + B11) +
    // A10 * (B00 + B11 - B01 + B01 - B11) +
    // A11 * (B00 + B11 - B01)
    // =
    // A10 * B00 + A11 * (B00 + B11 - B01)
    ikj_matmul_asm(&M3[0][0], &T0[0][0], &T1[0][0], &matrix_dim);

    // A00 * B00 + A01 * B10
    ikj_matmul_asm(Strassen_C00, Strassen_A01, Strassen_B10, &matrix_dim);

    // B01 + B10 - B00 - B11
    matrix_sub_acc(&T1[0][0], Strassen_B10, Strassen_B00, dim, dim);

    // A10 * B00 + A11 * (B00 + B11 - B01) + A11 * (B01 + B10 - B00 - B11)
    // =
    // A10 * B00 + A11 * B10
    matrix_add(Strassen_C10, Strassen_C10, &M3[0][0], dim, dim);
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

    matrix_add(&T2[0][0], Strassen_A10, Strassen_A11, dim, dim);
    matrix_sub(&T3[0][0], Strassen_B01, Strassen_B00, dim, dim);

    for(size_t i = 0; i < dim; i++){
        for(size_t j = 0; j < dim; j++){
            T0[i][j] = 0;
        }
    }
    // (A10 + A11) * (B01 - B00)
    ikj_matmul_asm(&T0[0][0], &T2[0][0], &T3[0][0], &matrix_dim);

    // A00 * B01 + A01 * B11 + A10 * (B00 - B01) + A11 * (B00 - B01) +
    // (A10 + A11) * (B01 - B00)
    // =
    // A00 * B01 + A01 * B11
    matrix_add(Strassen_C01, Strassen_C01, &T0[0][0], dim, dim);

    // A10 * B00 + A11 * (B00 + B11 - B01) + (A10 + A11) * (B01 - B00)
    // =
    // A10 * B01 + A11 * B11
    matrix_add_acc(Strassen_C11, &M3[0][0], &T0[0][0], dim, dim);



}











