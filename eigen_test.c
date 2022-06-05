

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <arm_neon.h>

#include "m1cycles.h"

#include "params.h"

#define BLOCK8
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

extern "C" void ijk_matmla_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
extern void ijk_matmla_asm(int16_t*, int16_t*, int16_t*, struct dimension*);
#endif

void ikj_matmul_SIMD(int16_t des[DENSE_DIM_I][DENSE_DIM_J], int16_t src1[DENSE_DIM_I][DENSE_DIM_K], int16_t src2[DENSE_DIM_K][DENSE_DIM_J]);
void ijk_matmla_SIMD(int16_t des[DENSE_DIM_I][DENSE_DIM_J], int16_t src1[DENSE_DIM_I][DENSE_DIM_K], int16_t src2[DENSE_DIM_K][DENSE_DIM_J]);


int main(void){

    long long start, end;
    long long ns;

    int nonZeros;

// ================================
// Dense settings

    int16_t N[DENSE_DIM_I][DENSE_DIM_J];
    int16_t A[DENSE_DIM_I][DENSE_DIM_J];
    int16_t A0[DENSE_DIM_I][DENSE_DIM_K];
    int16_t A1[DENSE_DIM_K][DENSE_DIM_J];

    Eigen::Matrix<int16_t, DENSE_DIM_I, DENSE_DIM_J> M;
    Eigen::Matrix<int16_t, DENSE_DIM_I, DENSE_DIM_K> M0 = Eigen::Matrix<int16_t, DENSE_DIM_I, DENSE_DIM_K>::Random();
    Eigen::Matrix<int16_t, DENSE_DIM_K, DENSE_DIM_J> M1 = Eigen::Matrix<int16_t, DENSE_DIM_K, DENSE_DIM_J>::Random();


    for(size_t i = 0; i < DENSE_DIM_I; i++){
        for(size_t j = 0; j < DENSE_DIM_J; j++){
            N[i][j] = A[i][j] = M(i, j) = 0;
        }
    }

    for(size_t i = 0; i < DENSE_DIM_I; i++){
        for(size_t k = 0; k < DENSE_DIM_K; k++){
            A0[i][k] = M0(i, k);
        }
    }

    for(size_t k = 0; k < DENSE_DIM_K; k++){
        for(size_t j = 0; j < DENSE_DIM_J; j++){
            A1[k][j] = M1(k, j);
        }
    }

// ================================
// Sparse settings

    Eigen::SparseMatrix<int16_t> SparseM(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t> CondenseM(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t> SparseM0(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t, Eigen::RowMajor> SparseM1(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t> CondenseM00(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t> CondenseM01(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t, Eigen::RowMajor> CondenseM10(SPARSE_DIM, SPARSE_DIM);
    Eigen::SparseMatrix<int16_t, Eigen::RowMajor> CondenseM11(SPARSE_DIM, SPARSE_DIM);

    std::vector<Eigen::Triplet<int16_t>> SparseCol;
    std::vector<Eigen::Triplet<int16_t>> CondenseCol0;
    std::vector<Eigen::Triplet<int16_t>> CondenseCol1;
    std::vector<Eigen::Triplet<int16_t>> SparseRow;
    std::vector<Eigen::Triplet<int16_t>> CondenseRow0;
    std::vector<Eigen::Triplet<int16_t>> CondenseRow1;

    std::vector<int16_t> buff0;
    std::vector<int16_t> buff1;

    for(int i = 0; i < SPARSE_DIM; i++){
        for(int j = 0; j < SPARSE_DIM; j++){
            if(j < DENSE_DIM){
                if( (rand() & 0x3f) == 0 ){
                    SparseCol.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                    CondenseCol0.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                }
            }else{
                if( (rand() & 0x3ff) == 0 ){
                    SparseCol.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                    CondenseCol1.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                }
            }
        }
    }

    for(int i = 0; i < SPARSE_DIM; i++){
        for(int j = 0; j < SPARSE_DIM; j++){
            if(i < DENSE_DIM){
                if( (rand() & 0x3f) == 0 ){
                    SparseRow.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                    CondenseRow0.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                }
            }else{
                if( (rand() & 0x3ff) == 0 ){
                    SparseRow.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                    CondenseRow1.push_back(Eigen::Triplet<int16_t>(i, j, i * SPARSE_DIM + j));
                }
            }
        }
    }

    SparseM0.setFromTriplets(SparseCol.begin(), SparseCol.end());
    SparseM1.setFromTriplets(SparseRow.begin(), SparseRow.end());

    CondenseM00.setFromTriplets(CondenseCol0.begin(), CondenseCol0.end());
    CondenseM01.setFromTriplets(CondenseCol1.begin(), CondenseCol1.end());
    CondenseM10.setFromTriplets(CondenseRow0.begin(), CondenseRow0.end());
    CondenseM11.setFromTriplets(CondenseRow1.begin(), CondenseRow1.end());

// ================================
// Dense

    setup_rdtsc();

    matrix_dim.dim_i = DENSE_DIM_I;
    matrix_dim.dim_k = DENSE_DIM_K;
    matrix_dim.dim_j = DENSE_DIM_J;

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ikj_matmul_asm(&A[0][0], &A0[0][0], &A1[0][0], &matrix_dim);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ikj SIMD asm Dense cycles:\n%lld\n", ns);

// ================================

    for(size_t i = 0; i < DENSE_DIM_I; i++){
        for(size_t j = 0; j < DENSE_DIM_J; j++){
            N[i][j] = 0;
        }
    }

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ikj_matmul_SIMD(N, A0, A1);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ikj SIMD intrinsics Dense cycles:\n%lld\n", ns);

#ifdef TEST
    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            if(A[i][j] != N[i][j]){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    A[i][j], N[i][j]);
            }
        }
    }
#endif

// ================================

    for(size_t i = 0; i < DENSE_DIM_I; i++){
        for(size_t j = 0; j < DENSE_DIM_J; j++){
            N[i][j] = 0;
        }
    }

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ijk_matmla_asm(&N[0][0], &A0[0][0], &A1[0][0], &matrix_dim);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ijk SIMD asm Dense cycles:\n%lld\n", ns);

#ifdef TEST
    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            if(A[i][j] != N[i][j]){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    A[i][j], N[i][j]);
            }
        }
    }
#endif

// ================================

    for(size_t i = 0; i < DENSE_DIM_I; i++){
        for(size_t j = 0; j < DENSE_DIM_J; j++){
            N[i][j] = 0;
        }
    }

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        ijk_matmla_SIMD(N, A0, A1);
    }
    end = rdtsc();
    ns = (end - start);
    printf("ijk SIMD intrinsics Dense cycles:\n%lld\n", ns);

#ifdef TEST
    for(size_t i = 0; i < SQUARE_DIM; i++){
        for(size_t j = 0; j < SQUARE_DIM; j++){
            if(A[i][j] != N[i][j]){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    A[i][j], N[i][j]);
            }
        }
    }
#endif

// ================================

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        M += M0 * M1;
    }
    end = rdtsc();
    ns = (end - start);
    printf("Eigen Dense cycles:\n%lld\n", ns);

#ifdef TEST
    for(size_t i = 0; i < DENSE_DIM_I; i++){
        for(size_t j = 0; j < DENSE_DIM_J; j++){
            if(A[i][j] != M(i, j)){
                fprintf(stderr, "%4zu, %4zu: %8d, %8d\n", i, j,
                    A[i][j], M(i, j));
            }
        }
    }
#endif

// ================================
// Sparse

    printf("non-zeros: %d, %d\n", SparseM0.nonZeros(), SparseM1.nonZeros());

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        SparseM += SparseM0 * SparseM1;
    }
    end = rdtsc();
    ns = (end - start);
    printf("Eigen Sparse cycles:\n%lld\n", ns);

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        CondenseM += CondenseM00 * CondenseM10;
    }
    end = rdtsc();
    ns = (end - start);
    printf("Eigen Condensed Sparse cycles:\n%lld\n", ns);

    start = rdtsc();
    for(size_t i = 0; i < 16; i++){
        CondenseM += CondenseM01 * CondenseM11;
    }
    end = rdtsc();
    ns = (end - start);
    printf("Eigen Sparse Sparse cycles:\n%lld\n", ns);

#ifdef TEST
    assert(SparseM.nonZeros() == CondenseM.nonZeros());

    for(int i = 0; i < SparseM.outerSize(); ++i){
        for(Eigen::SparseMatrix<int16_t>::InnerIterator iter(SparseM, i); iter; ++iter){
            buff0.push_back(iter.col());
            buff0.push_back(iter.row());
            buff0.push_back(iter.value());
        }
    }

    for(int i = 0; i < CondenseM.outerSize(); ++i){
        for(Eigen::SparseMatrix<int16_t>::InnerIterator iter(CondenseM, i); iter; ++iter){
            buff1.push_back(iter.col());
            buff1.push_back(iter.row());
            buff1.push_back(iter.value());
        }
    }

    for(size_t i = 0; i < SparseM.nonZeros() * 3; i++){
        assert(buff0[i] == buff1[i]);
    }
#endif




}


void ijk_matmla_SIMD(int16_t des[DENSE_DIM_I][DENSE_DIM_J], int16_t src1[DENSE_DIM_I][DENSE_DIM_K], int16_t src2[DENSE_DIM_K][DENSE_DIM_J]){

    int16x8_t t0;
    int16x8_t t1;
    int16x8_t t2;
    int16x8_t t3;
    int16x8_t t4;
    int16x8_t t5;
    int16x8_t t6;
    int16x8_t t7;
    int16x8_t t8;
    int16x8_t t9;
    int16x8_t t10;
    int16x8_t t11;

    int16x8_t acc0;
    int16x8_t acc1;
    int16x8_t acc2;
    int16x8_t acc3;
    int16x8_t acc4;
    int16x8_t acc5;
    int16x8_t acc6;
    int16x8_t acc7;
    int16x8_t acc8;
    int16x8_t acc9;
    int16x8_t acc10;
    int16x8_t acc11;

    int16x8_t v0;
    int16x8_t v1;
    int16x8_t v2;
    int16x8_t v3;
    int16x8_t v4;
    int16x8_t v5;
    int16x8_t v6;
    int16x8_t v7;

    for(size_t i = 0; i < DENSE_DIM_I; i += 12){
        for(size_t j = 0; j < DENSE_DIM_J; j += 8){

            acc0 = vld1q_s16(&des[i + 0][j]);
            acc1 = vld1q_s16(&des[i + 1][j]);
            acc2 = vld1q_s16(&des[i + 2][j]);
            acc3 = vld1q_s16(&des[i + 3][j]);
            acc4 = vld1q_s16(&des[i + 4][j]);
            acc5 = vld1q_s16(&des[i + 5][j]);
            acc6 = vld1q_s16(&des[i + 6][j]);
            acc7 = vld1q_s16(&des[i + 7][j]);
            acc8 = vld1q_s16(&des[i + 8][j]);
            acc9 = vld1q_s16(&des[i + 9][j]);
            acc10 = vld1q_s16(&des[i + 10][j]);
            acc11 = vld1q_s16(&des[i + 11][j]);

            for(size_t k = 0; k < DENSE_DIM_K; k += 8){

                t0 = vld1q_s16(&src1[i + 0][k]);
                t1 = vld1q_s16(&src1[i + 1][k]);
                t2 = vld1q_s16(&src1[i + 2][k]);
                t3 = vld1q_s16(&src1[i + 3][k]);
                t4 = vld1q_s16(&src1[i + 4][k]);
                t5 = vld1q_s16(&src1[i + 5][k]);
                t6 = vld1q_s16(&src1[i + 6][k]);
                t7 = vld1q_s16(&src1[i + 7][k]);
                t8 = vld1q_s16(&src1[i + 8][k]);
                t9 = vld1q_s16(&src1[i + 9][k]);
                t10 = vld1q_s16(&src1[i + 10][k]);
                t11 = vld1q_s16(&src1[i + 11][k]);

                v0 = vld1q_s16(&src2[k + 0][j]);
                v1 = vld1q_s16(&src2[k + 1][j]);
                v2 = vld1q_s16(&src2[k + 2][j]);
                v3 = vld1q_s16(&src2[k + 3][j]);
                v4 = vld1q_s16(&src2[k + 4][j]);
                v5 = vld1q_s16(&src2[k + 5][j]);
                v6 = vld1q_s16(&src2[k + 6][j]);
                v7 = vld1q_s16(&src2[k + 7][j]);

                acc0 = vmlaq_n_s16(acc0, v0, t0[0]);
                acc0 = vmlaq_n_s16(acc0, v1, t0[1]);
                acc0 = vmlaq_n_s16(acc0, v2, t0[2]);
                acc0 = vmlaq_n_s16(acc0, v3, t0[3]);
                acc0 = vmlaq_n_s16(acc0, v4, t0[4]);
                acc0 = vmlaq_n_s16(acc0, v5, t0[5]);
                acc0 = vmlaq_n_s16(acc0, v6, t0[6]);
                acc0 = vmlaq_n_s16(acc0, v7, t0[7]);

                acc1 = vmlaq_n_s16(acc1, v0, t1[0]);
                acc1 = vmlaq_n_s16(acc1, v1, t1[1]);
                acc1 = vmlaq_n_s16(acc1, v2, t1[2]);
                acc1 = vmlaq_n_s16(acc1, v3, t1[3]);
                acc1 = vmlaq_n_s16(acc1, v4, t1[4]);
                acc1 = vmlaq_n_s16(acc1, v5, t1[5]);
                acc1 = vmlaq_n_s16(acc1, v6, t1[6]);
                acc1 = vmlaq_n_s16(acc1, v7, t1[7]);

                acc2 = vmlaq_n_s16(acc2, v0, t2[0]);
                acc2 = vmlaq_n_s16(acc2, v1, t2[1]);
                acc2 = vmlaq_n_s16(acc2, v2, t2[2]);
                acc2 = vmlaq_n_s16(acc2, v3, t2[3]);
                acc2 = vmlaq_n_s16(acc2, v4, t2[4]);
                acc2 = vmlaq_n_s16(acc2, v5, t2[5]);
                acc2 = vmlaq_n_s16(acc2, v6, t2[6]);
                acc2 = vmlaq_n_s16(acc2, v7, t2[7]);

                acc3 = vmlaq_n_s16(acc3, v0, t3[0]);
                acc3 = vmlaq_n_s16(acc3, v1, t3[1]);
                acc3 = vmlaq_n_s16(acc3, v2, t3[2]);
                acc3 = vmlaq_n_s16(acc3, v3, t3[3]);
                acc3 = vmlaq_n_s16(acc3, v4, t3[4]);
                acc3 = vmlaq_n_s16(acc3, v5, t3[5]);
                acc3 = vmlaq_n_s16(acc3, v6, t3[6]);
                acc3 = vmlaq_n_s16(acc3, v7, t3[7]);

                acc4 = vmlaq_n_s16(acc4, v0, t4[0]);
                acc4 = vmlaq_n_s16(acc4, v1, t4[1]);
                acc4 = vmlaq_n_s16(acc4, v2, t4[2]);
                acc4 = vmlaq_n_s16(acc4, v3, t4[3]);
                acc4 = vmlaq_n_s16(acc4, v4, t4[4]);
                acc4 = vmlaq_n_s16(acc4, v5, t4[5]);
                acc4 = vmlaq_n_s16(acc4, v6, t4[6]);
                acc4 = vmlaq_n_s16(acc4, v7, t4[7]);

                acc5 = vmlaq_n_s16(acc5, v0, t5[0]);
                acc5 = vmlaq_n_s16(acc5, v1, t5[1]);
                acc5 = vmlaq_n_s16(acc5, v2, t5[2]);
                acc5 = vmlaq_n_s16(acc5, v3, t5[3]);
                acc5 = vmlaq_n_s16(acc5, v4, t5[4]);
                acc5 = vmlaq_n_s16(acc5, v5, t5[5]);
                acc5 = vmlaq_n_s16(acc5, v6, t5[6]);
                acc5 = vmlaq_n_s16(acc5, v7, t5[7]);

                acc6 = vmlaq_n_s16(acc6, v0, t6[0]);
                acc6 = vmlaq_n_s16(acc6, v1, t6[1]);
                acc6 = vmlaq_n_s16(acc6, v2, t6[2]);
                acc6 = vmlaq_n_s16(acc6, v3, t6[3]);
                acc6 = vmlaq_n_s16(acc6, v4, t6[4]);
                acc6 = vmlaq_n_s16(acc6, v5, t6[5]);
                acc6 = vmlaq_n_s16(acc6, v6, t6[6]);
                acc6 = vmlaq_n_s16(acc6, v7, t6[7]);

                acc7 = vmlaq_n_s16(acc7, v0, t7[0]);
                acc7 = vmlaq_n_s16(acc7, v1, t7[1]);
                acc7 = vmlaq_n_s16(acc7, v2, t7[2]);
                acc7 = vmlaq_n_s16(acc7, v3, t7[3]);
                acc7 = vmlaq_n_s16(acc7, v4, t7[4]);
                acc7 = vmlaq_n_s16(acc7, v5, t7[5]);
                acc7 = vmlaq_n_s16(acc7, v6, t7[6]);
                acc7 = vmlaq_n_s16(acc7, v7, t7[7]);

                acc8 = vmlaq_n_s16(acc8, v0, t8[0]);
                acc8 = vmlaq_n_s16(acc8, v1, t8[1]);
                acc8 = vmlaq_n_s16(acc8, v2, t8[2]);
                acc8 = vmlaq_n_s16(acc8, v3, t8[3]);
                acc8 = vmlaq_n_s16(acc8, v4, t8[4]);
                acc8 = vmlaq_n_s16(acc8, v5, t8[5]);
                acc8 = vmlaq_n_s16(acc8, v6, t8[6]);
                acc8 = vmlaq_n_s16(acc8, v7, t8[7]);

                acc9 = vmlaq_n_s16(acc9, v0, t9[0]);
                acc9 = vmlaq_n_s16(acc9, v1, t9[1]);
                acc9 = vmlaq_n_s16(acc9, v2, t9[2]);
                acc9 = vmlaq_n_s16(acc9, v3, t9[3]);
                acc9 = vmlaq_n_s16(acc9, v4, t9[4]);
                acc9 = vmlaq_n_s16(acc9, v5, t9[5]);
                acc9 = vmlaq_n_s16(acc9, v6, t9[6]);
                acc9 = vmlaq_n_s16(acc9, v7, t9[7]);

                acc10 = vmlaq_n_s16(acc10, v0, t10[0]);
                acc10 = vmlaq_n_s16(acc10, v1, t10[1]);
                acc10 = vmlaq_n_s16(acc10, v2, t10[2]);
                acc10 = vmlaq_n_s16(acc10, v3, t10[3]);
                acc10 = vmlaq_n_s16(acc10, v4, t10[4]);
                acc10 = vmlaq_n_s16(acc10, v5, t10[5]);
                acc10 = vmlaq_n_s16(acc10, v6, t10[6]);
                acc10 = vmlaq_n_s16(acc10, v7, t10[7]);

                acc11 = vmlaq_n_s16(acc11, v0, t11[0]);
                acc11 = vmlaq_n_s16(acc11, v1, t11[1]);
                acc11 = vmlaq_n_s16(acc11, v2, t11[2]);
                acc11 = vmlaq_n_s16(acc11, v3, t11[3]);
                acc11 = vmlaq_n_s16(acc11, v4, t11[4]);
                acc11 = vmlaq_n_s16(acc11, v5, t11[5]);
                acc11 = vmlaq_n_s16(acc11, v6, t11[6]);
                acc11 = vmlaq_n_s16(acc11, v7, t11[7]);


            }

            vst1q_s16(&des[i + 0][j], acc0);
            vst1q_s16(&des[i + 1][j], acc1);
            vst1q_s16(&des[i + 2][j], acc2);
            vst1q_s16(&des[i + 3][j], acc3);
            vst1q_s16(&des[i + 4][j], acc4);
            vst1q_s16(&des[i + 5][j], acc5);
            vst1q_s16(&des[i + 6][j], acc6);
            vst1q_s16(&des[i + 7][j], acc7);
            vst1q_s16(&des[i + 8][j], acc8);
            vst1q_s16(&des[i + 9][j], acc9);
            vst1q_s16(&des[i + 10][j], acc10);
            vst1q_s16(&des[i + 11][j], acc11);

        }
    }


}


void ikj_matmul_SIMD(int16_t des[DENSE_DIM_I][DENSE_DIM_J], int16_t src1[DENSE_DIM_I][DENSE_DIM_K], int16_t src2[DENSE_DIM_K][DENSE_DIM_J]){

    int16x8_t t0;
    int16x8_t t1;
    int16x8_t t2;
    int16x8_t t3;
    int16x8_t t4;
    int16x8_t t5;
    int16x8_t t6;
    int16x8_t t7;

    int16x8_t acc0;
    int16x8_t acc1;
    int16x8_t acc2;
    int16x8_t acc3;
    int16x8_t acc4;
    int16x8_t acc5;
    int16x8_t acc6;
    int16x8_t acc7;

    int16x8_t v0;
    int16x8_t v1;
    int16x8_t v2;
    int16x8_t v3;
    int16x8_t v4;
    int16x8_t v5;
    int16x8_t v6;
    int16x8_t v7;

#if defined(BLOCK4)
    for(size_t i = 0; i < DENSE_DIM_I; i += 4){
        for(size_t k = 0; k < DENSE_DIM_K; k += 8){

            t0 = vld1q_s16(&src1[i + 0][k + 0]);
            t1 = vld1q_s16(&src1[i + 1][k + 0]);
            t2 = vld1q_s16(&src1[i + 2][k + 0]);
            t3 = vld1q_s16(&src1[i + 3][k + 0]);

            for(size_t j = 0; j < DENSE_DIM_J; j += 8){

                acc0 = vld1q_s16(&des[i + 0][j]);
                acc1 = vld1q_s16(&des[i + 1][j]);
                acc2 = vld1q_s16(&des[i + 2][j]);
                acc3 = vld1q_s16(&des[i + 3][j]);

                v0 = vld1q_s16(&src2[k + 0][j]);
                v1 = vld1q_s16(&src2[k + 1][j]);
                v2 = vld1q_s16(&src2[k + 2][j]);
                v3 = vld1q_s16(&src2[k + 3][j]);
                v4 = vld1q_s16(&src2[k + 4][j]);
                v5 = vld1q_s16(&src2[k + 5][j]);
                v6 = vld1q_s16(&src2[k + 6][j]);
                v7 = vld1q_s16(&src2[k + 7][j]);

                acc0 = vmlaq_n_s16(acc0, v0, t0[0]);
                acc0 = vmlaq_n_s16(acc0, v1, t0[1]);
                acc0 = vmlaq_n_s16(acc0, v2, t0[2]);
                acc0 = vmlaq_n_s16(acc0, v3, t0[3]);
                acc0 = vmlaq_n_s16(acc0, v4, t0[4]);
                acc0 = vmlaq_n_s16(acc0, v5, t0[5]);
                acc0 = vmlaq_n_s16(acc0, v6, t0[6]);
                acc0 = vmlaq_n_s16(acc0, v7, t0[7]);

                acc1 = vmlaq_n_s16(acc1, v0, t1[0]);
                acc1 = vmlaq_n_s16(acc1, v1, t1[1]);
                acc1 = vmlaq_n_s16(acc1, v2, t1[2]);
                acc1 = vmlaq_n_s16(acc1, v3, t1[3]);
                acc1 = vmlaq_n_s16(acc1, v4, t1[4]);
                acc1 = vmlaq_n_s16(acc1, v5, t1[5]);
                acc1 = vmlaq_n_s16(acc1, v6, t1[6]);
                acc1 = vmlaq_n_s16(acc1, v7, t1[7]);

                acc2 = vmlaq_n_s16(acc2, v0, t2[0]);
                acc2 = vmlaq_n_s16(acc2, v1, t2[1]);
                acc2 = vmlaq_n_s16(acc2, v2, t2[2]);
                acc2 = vmlaq_n_s16(acc2, v3, t2[3]);
                acc2 = vmlaq_n_s16(acc2, v4, t2[4]);
                acc2 = vmlaq_n_s16(acc2, v5, t2[5]);
                acc2 = vmlaq_n_s16(acc2, v6, t2[6]);
                acc2 = vmlaq_n_s16(acc2, v7, t2[7]);

                acc3 = vmlaq_n_s16(acc3, v0, t3[0]);
                acc3 = vmlaq_n_s16(acc3, v1, t3[1]);
                acc3 = vmlaq_n_s16(acc3, v2, t3[2]);
                acc3 = vmlaq_n_s16(acc3, v3, t3[3]);
                acc3 = vmlaq_n_s16(acc3, v4, t3[4]);
                acc3 = vmlaq_n_s16(acc3, v5, t3[5]);
                acc3 = vmlaq_n_s16(acc3, v6, t3[6]);
                acc3 = vmlaq_n_s16(acc3, v7, t3[7]);

                vst1q_s16(&des[i + 0][j], acc0);
                vst1q_s16(&des[i + 1][j], acc1);
                vst1q_s16(&des[i + 2][j], acc2);
                vst1q_s16(&des[i + 3][j], acc3);

            }
        }
    }

#else

    for(size_t i = 0; i < DENSE_DIM_I; i += 8){
        for(size_t k = 0; k < DENSE_DIM_K; k += 8){

            t0 = vld1q_s16(&src1[i + 0][k + 0]);
            t1 = vld1q_s16(&src1[i + 1][k + 0]);
            t2 = vld1q_s16(&src1[i + 2][k + 0]);
            t3 = vld1q_s16(&src1[i + 3][k + 0]);
            t4 = vld1q_s16(&src1[i + 4][k + 0]);
            t5 = vld1q_s16(&src1[i + 5][k + 0]);
            t6 = vld1q_s16(&src1[i + 6][k + 0]);
            t7 = vld1q_s16(&src1[i + 7][k + 0]);

            for(size_t j = 0; j < DENSE_DIM_J; j += 8){

                acc0 = vld1q_s16(&des[i + 0][j]);
                acc1 = vld1q_s16(&des[i + 1][j]);
                acc2 = vld1q_s16(&des[i + 2][j]);
                acc3 = vld1q_s16(&des[i + 3][j]);
                acc4 = vld1q_s16(&des[i + 4][j]);
                acc5 = vld1q_s16(&des[i + 5][j]);
                acc6 = vld1q_s16(&des[i + 6][j]);
                acc7 = vld1q_s16(&des[i + 7][j]);

                v0 = vld1q_s16(&src2[k + 0][j]);
                v1 = vld1q_s16(&src2[k + 1][j]);
                v2 = vld1q_s16(&src2[k + 2][j]);
                v3 = vld1q_s16(&src2[k + 3][j]);
                v4 = vld1q_s16(&src2[k + 4][j]);
                v5 = vld1q_s16(&src2[k + 5][j]);
                v6 = vld1q_s16(&src2[k + 6][j]);
                v7 = vld1q_s16(&src2[k + 7][j]);

                acc0 = vmlaq_n_s16(acc0, v0, t0[0]);
                acc0 = vmlaq_n_s16(acc0, v1, t0[1]);
                acc0 = vmlaq_n_s16(acc0, v2, t0[2]);
                acc0 = vmlaq_n_s16(acc0, v3, t0[3]);
                acc0 = vmlaq_n_s16(acc0, v4, t0[4]);
                acc0 = vmlaq_n_s16(acc0, v5, t0[5]);
                acc0 = vmlaq_n_s16(acc0, v6, t0[6]);
                acc0 = vmlaq_n_s16(acc0, v7, t0[7]);

                acc1 = vmlaq_n_s16(acc1, v0, t1[0]);
                acc1 = vmlaq_n_s16(acc1, v1, t1[1]);
                acc1 = vmlaq_n_s16(acc1, v2, t1[2]);
                acc1 = vmlaq_n_s16(acc1, v3, t1[3]);
                acc1 = vmlaq_n_s16(acc1, v4, t1[4]);
                acc1 = vmlaq_n_s16(acc1, v5, t1[5]);
                acc1 = vmlaq_n_s16(acc1, v6, t1[6]);
                acc1 = vmlaq_n_s16(acc1, v7, t1[7]);

                acc2 = vmlaq_n_s16(acc2, v0, t2[0]);
                acc2 = vmlaq_n_s16(acc2, v1, t2[1]);
                acc2 = vmlaq_n_s16(acc2, v2, t2[2]);
                acc2 = vmlaq_n_s16(acc2, v3, t2[3]);
                acc2 = vmlaq_n_s16(acc2, v4, t2[4]);
                acc2 = vmlaq_n_s16(acc2, v5, t2[5]);
                acc2 = vmlaq_n_s16(acc2, v6, t2[6]);
                acc2 = vmlaq_n_s16(acc2, v7, t2[7]);

                acc3 = vmlaq_n_s16(acc3, v0, t3[0]);
                acc3 = vmlaq_n_s16(acc3, v1, t3[1]);
                acc3 = vmlaq_n_s16(acc3, v2, t3[2]);
                acc3 = vmlaq_n_s16(acc3, v3, t3[3]);
                acc3 = vmlaq_n_s16(acc3, v4, t3[4]);
                acc3 = vmlaq_n_s16(acc3, v5, t3[5]);
                acc3 = vmlaq_n_s16(acc3, v6, t3[6]);
                acc3 = vmlaq_n_s16(acc3, v7, t3[7]);

                acc4 = vmlaq_n_s16(acc4, v0, t4[0]);
                acc4 = vmlaq_n_s16(acc4, v1, t4[1]);
                acc4 = vmlaq_n_s16(acc4, v2, t4[2]);
                acc4 = vmlaq_n_s16(acc4, v3, t4[3]);
                acc4 = vmlaq_n_s16(acc4, v4, t4[4]);
                acc4 = vmlaq_n_s16(acc4, v5, t4[5]);
                acc4 = vmlaq_n_s16(acc4, v6, t4[6]);
                acc4 = vmlaq_n_s16(acc4, v7, t4[7]);

                acc5 = vmlaq_n_s16(acc5, v0, t5[0]);
                acc5 = vmlaq_n_s16(acc5, v1, t5[1]);
                acc5 = vmlaq_n_s16(acc5, v2, t5[2]);
                acc5 = vmlaq_n_s16(acc5, v3, t5[3]);
                acc5 = vmlaq_n_s16(acc5, v4, t5[4]);
                acc5 = vmlaq_n_s16(acc5, v5, t5[5]);
                acc5 = vmlaq_n_s16(acc5, v6, t5[6]);
                acc5 = vmlaq_n_s16(acc5, v7, t5[7]);

                acc6 = vmlaq_n_s16(acc6, v0, t6[0]);
                acc6 = vmlaq_n_s16(acc6, v1, t6[1]);
                acc6 = vmlaq_n_s16(acc6, v2, t6[2]);
                acc6 = vmlaq_n_s16(acc6, v3, t6[3]);
                acc6 = vmlaq_n_s16(acc6, v4, t6[4]);
                acc6 = vmlaq_n_s16(acc6, v5, t6[5]);
                acc6 = vmlaq_n_s16(acc6, v6, t6[6]);
                acc6 = vmlaq_n_s16(acc6, v7, t6[7]);

                acc7 = vmlaq_n_s16(acc7, v0, t7[0]);
                acc7 = vmlaq_n_s16(acc7, v1, t7[1]);
                acc7 = vmlaq_n_s16(acc7, v2, t7[2]);
                acc7 = vmlaq_n_s16(acc7, v3, t7[3]);
                acc7 = vmlaq_n_s16(acc7, v4, t7[4]);
                acc7 = vmlaq_n_s16(acc7, v5, t7[5]);
                acc7 = vmlaq_n_s16(acc7, v6, t7[6]);
                acc7 = vmlaq_n_s16(acc7, v7, t7[7]);

                vst1q_s16(&des[i + 0][j], acc0);
                vst1q_s16(&des[i + 1][j], acc1);
                vst1q_s16(&des[i + 2][j], acc2);
                vst1q_s16(&des[i + 3][j], acc3);
                vst1q_s16(&des[i + 4][j], acc4);
                vst1q_s16(&des[i + 5][j], acc5);
                vst1q_s16(&des[i + 6][j], acc6);
                vst1q_s16(&des[i + 7][j], acc7);

            }
        }
    }

#endif



}
















