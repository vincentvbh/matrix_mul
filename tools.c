
#include "tools.h"

#include <arm_neon.h>

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

void matrix_add_fromM(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, int16_t *M2, size_t jump_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;

    for(size_t i = 0; i < dim_i; i++){
        for(size_t j = 0; j < dim_j; j += 24){

            a0 = vld1q_s16(M1 + i * jump_j + j + 0 * 8);
            a1 = vld1q_s16(M1 + i * jump_j + j + 1 * 8);
            a2 = vld1q_s16(M1 + i * jump_j + j + 2 * 8);

            vst1q_s16(src1 + i * dim_j + j + 0 * 8, a0);
            vst1q_s16(src1 + i * dim_j + j + 1 * 8, a1);
            vst1q_s16(src1 + i * dim_j + j + 2 * 8, a2);

            b0 = vld1q_s16(M2 + i * jump_j + j + 0 * 8);
            b1 = vld1q_s16(M2 + i * jump_j + j + 1 * 8);
            b2 = vld1q_s16(M2 + i * jump_j + j + 2 * 8);

            vst1q_s16(src2 + i * dim_j + j + 0 * 8, b0);
            vst1q_s16(src2 + i * dim_j + j + 1 * 8, b1);
            vst1q_s16(src2 + i * dim_j + j + 2 * 8, b2);

            vst1q_s16(des + i * dim_j + j + 0 * 8, a0 + b0);
            vst1q_s16(des + i * dim_j + j + 1 * 8, a1 + b1);
            vst1q_s16(des + i * dim_j + j + 2 * 8, a2 + b2);

        }
    }

}

void matrix_addx2(
    int16_t *des1, int16_t *des2,
    int16_t *src1, int16_t *src2,
    int16_t *addend,
    size_t dim_i, size_t dim_j){

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

    int16x8_t t0;
    int16x8_t t1;
    int16x8_t t2;
    int16x8_t t3;
    int16x8_t t4;
    int16x8_t t5;

    for(size_t i = 0; i < (dim_i * dim_j); i += 48){

        t0 = vld1q_s16(addend + i + 0 * 8);
        t1 = vld1q_s16(addend + i + 1 * 8);
        t2 = vld1q_s16(addend + i + 2 * 8);
        t3 = vld1q_s16(addend + i + 3 * 8);
        t4 = vld1q_s16(addend + i + 4 * 8);
        t5 = vld1q_s16(addend + i + 5 * 8);

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

        vst1q_s16(des1 + i + 0 * 8, a0 + t0);
        vst1q_s16(des1 + i + 1 * 8, a1 + t1);
        vst1q_s16(des1 + i + 2 * 8, a2 + t2);
        vst1q_s16(des1 + i + 3 * 8, a3 + t3);
        vst1q_s16(des1 + i + 4 * 8, a4 + t4);
        vst1q_s16(des1 + i + 5 * 8, a5 + t5);

        vst1q_s16(des2 + i + 0 * 8, b0 + t0);
        vst1q_s16(des2 + i + 1 * 8, b1 + t1);
        vst1q_s16(des2 + i + 2 * 8, b2 + t2);
        vst1q_s16(des2 + i + 3 * 8, b3 + t3);
        vst1q_s16(des2 + i + 4 * 8, b4 + t4);
        vst1q_s16(des2 + i + 5 * 8, b5 + t5);

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

void matrix_sub_fromM(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, int16_t *M2, size_t jump_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;

    for(size_t i = 0; i < dim_i; i++){
        for(size_t j = 0; j < dim_j; j += 24){

            a0 = vld1q_s16(M1 + i * jump_j + j + 0 * 8);
            a1 = vld1q_s16(M1 + i * jump_j + j + 1 * 8);
            a2 = vld1q_s16(M1 + i * jump_j + j + 2 * 8);

            vst1q_s16(src1 + i * dim_j + j + 0 * 8, a0);
            vst1q_s16(src1 + i * dim_j + j + 1 * 8, a1);
            vst1q_s16(src1 + i * dim_j + j + 2 * 8, a2);

            b0 = vld1q_s16(M2 + i * jump_j + j + 0 * 8);
            b1 = vld1q_s16(M2 + i * jump_j + j + 1 * 8);
            b2 = vld1q_s16(M2 + i * jump_j + j + 2 * 8);

            vst1q_s16(src2 + i * dim_j + j + 0 * 8, b0);
            vst1q_s16(src2 + i * dim_j + j + 1 * 8, b1);
            vst1q_s16(src2 + i * dim_j + j + 2 * 8, b2);

            vst1q_s16(des + i * dim_j + j + 0 * 8, a0 - b0);
            vst1q_s16(des + i * dim_j + j + 1 * 8, a1 - b1);
            vst1q_s16(des + i * dim_j + j + 2 * 8, a2 - b2);

        }
    }

}

void matrix_sub_fromM2(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M2, size_t jump_j){

    int16x8_t a0;
    int16x8_t a1;
    int16x8_t a2;

    int16x8_t b0;
    int16x8_t b1;
    int16x8_t b2;

    for(size_t i = 0; i < dim_i; i++){
        for(size_t j = 0; j < dim_j; j += 24){

            a0 = vld1q_s16(src1 + i * dim_j + j + 0 * 8);
            a1 = vld1q_s16(src1 + i * dim_j + j + 1 * 8);
            a2 = vld1q_s16(src1 + i * dim_j + j + 2 * 8);

            b0 = vld1q_s16(M2 + i * jump_j + j + 0 * 8);
            b1 = vld1q_s16(M2 + i * jump_j + j + 1 * 8);
            b2 = vld1q_s16(M2 + i * jump_j + j + 2 * 8);

            vst1q_s16(src2 + i * dim_j + j + 0 * 8, b0);
            vst1q_s16(src2 + i * dim_j + j + 1 * 8, b1);
            vst1q_s16(src2 + i * dim_j + j + 2 * 8, b2);

            vst1q_s16(des + i * dim_j + j + 0 * 8, a0 - b0);
            vst1q_s16(des + i * dim_j + j + 1 * 8, a1 - b1);
            vst1q_s16(des + i * dim_j + j + 2 * 8, a2 - b2);

        }
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

void matrix_sub_acc_fromM1(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, size_t jump_j){

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

    for(size_t i = 0; i < dim_i; i++){
        for(size_t j = 0; j < dim_j; j += 24){

            a0 = vld1q_s16(M1 + i * jump_j + j + 0 * 8);
            a1 = vld1q_s16(M1 + i * jump_j + j + 1 * 8);
            a2 = vld1q_s16(M1 + i * jump_j + j + 2 * 8);

            vst1q_s16(src1 + i * dim_j + j + 0 * 8, a0);
            vst1q_s16(src1 + i * dim_j + j + 1 * 8, a1);
            vst1q_s16(src1 + i * dim_j + j + 2 * 8, a2);

            c0 = vld1q_s16(des + i * dim_j + j + 0 * 8);
            c1 = vld1q_s16(des + i * dim_j + j + 1 * 8);
            c2 = vld1q_s16(des + i * dim_j + j + 2 * 8);

            b0 = vld1q_s16(src2 + i * dim_j + j + 0 * 8);
            b1 = vld1q_s16(src2 + i * dim_j + j + 1 * 8);
            b2 = vld1q_s16(src2 + i * dim_j + j + 2 * 8);

            vst1q_s16(des + i * dim_j + j + 0 * 8, a0 - b0 + c0);
            vst1q_s16(des + i * dim_j + j + 1 * 8, a1 - b1 + c1);
            vst1q_s16(des + i * dim_j + j + 2 * 8, a2 - b2 + c2);

        }
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

void matrix_sub_negacc_fromM1(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, size_t jump_j){

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

    for(size_t i = 0; i < dim_i; i++){
        for(size_t j = 0; j < dim_j; j += 24){

            a0 = vld1q_s16(M1 + i * jump_j + j + 0 * 8);
            a1 = vld1q_s16(M1 + i * jump_j + j + 1 * 8);
            a2 = vld1q_s16(M1 + i * jump_j + j + 2 * 8);

            vst1q_s16(src1 + i * dim_j + j + 0 * 8, a0);
            vst1q_s16(src1 + i * dim_j + j + 1 * 8, a1);
            vst1q_s16(src1 + i * dim_j + j + 2 * 8, a2);

            c0 = vld1q_s16(des + i * dim_j + j + 0 * 8);
            c1 = vld1q_s16(des + i * dim_j + j + 1 * 8);
            c2 = vld1q_s16(des + i * dim_j + j + 2 * 8);

            b0 = vld1q_s16(src2 + i * dim_j + j + 0 * 8);
            b1 = vld1q_s16(src2 + i * dim_j + j + 1 * 8);
            b2 = vld1q_s16(src2 + i * dim_j + j + 2 * 8);

            vst1q_s16(des + i * dim_j + j + 0 * 8, a0 - b0 - c0);
            vst1q_s16(des + i * dim_j + j + 1 * 8, a1 - b1 - c1);
            vst1q_s16(des + i * dim_j + j + 2 * 8, a2 - b2 - c2);

        }
    }

}

