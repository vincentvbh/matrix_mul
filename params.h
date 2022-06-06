#ifndef PARAMS_H
#define PARAMS_H

// Strassen's

// multiple of 48
// largest size for Apple M1: 48 * 15
#define SQUARE_DIM (48*5)

// compare against Eigen for Dense

// multiple of 12
#define DENSE_DIM_I (96)
// multiple of 8
#define DENSE_DIM_K (144)
// multiple of 8
#define DENSE_DIM_J (120)

// Eigen for Sparse
#define DENSE_DIM 120
#define SPARSE_DIM 1536



#endif


