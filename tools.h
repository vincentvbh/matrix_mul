#ifndef TOOLS_H
#define TOOLS_H

#include <stdint.h>
#include <stddef.h>

void matrix_add(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j);

void matrix_add_fromM(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, int16_t *M2, size_t jump_j);


void matrix_addx2(
    int16_t *des1, int16_t *des2,
    int16_t *src1, int16_t *src2,
    int16_t *addend,
    size_t dim_i, size_t dim_j);


void matrix_sub(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j);

void matrix_sub_fromM(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, int16_t *M2, size_t jump_j);

void matrix_sub_fromM2(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M2, size_t jump_j);

void matrix_add_acc(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j);

void matrix_sub_acc(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j);

void matrix_sub_acc_fromM1(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, size_t jump_j);

void matrix_sub_negacc(int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j);

void matrix_sub_negacc_fromM1(
    int16_t *des, int16_t *src1, int16_t *src2, size_t dim_i, size_t dim_j,
    int16_t *M1, size_t jump_j);

#endif








