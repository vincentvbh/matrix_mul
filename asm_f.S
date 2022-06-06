

.macro push_all

    sub sp, sp, #(16*9)
    stp x19, x20, [sp, #16*0]
    stp x21, x22, [sp, #16*1]
    stp x23, x24, [sp, #16*2]
    stp x25, x26, [sp, #16*3]
    stp x27, x28, [sp, #16*4]
    stp  d8,  d9, [sp, #16*5]
    stp d10, d11, [sp, #16*6]
    stp d12, d13, [sp, #16*7]
    stp d14, d15, [sp, #16*8]

.endm

.macro pop_all

    ldp x19, x20, [sp, #16*0]
    ldp x21, x22, [sp, #16*1]
    ldp x23, x24, [sp, #16*2]
    ldp x25, x26, [sp, #16*3]
    ldp x27, x28, [sp, #16*4]
    ldp  d8,  d9, [sp, #16*5]
    ldp d10, d11, [sp, #16*6]
    ldp d12, d13, [sp, #16*7]
    ldp d14, d15, [sp, #16*8]
    add sp, sp, #(16*9)

.endm



.align 8
.global ikj_matmul_asm
.global _ikj_matmul_asm
ikj_matmul_asm:
_ikj_matmul_asm:

    push_all

//    for(size_t i = 0; i < DIM_I; i++){
//        for(size_t k = 0; k < DIM_K; k++){
//            for(size_t j = 0; j < DIM_J; j++){
//                des[i][j] += src1[i][k] * src2[k][j];
//            }
//        }
//    }

    ldr w7, [x3, #0]
    ldr w11, [x3, #4]
    ldr w15, [x3, #8]

// ================

    _ikj_loopi:

// ================

    mov w9, w11
    _ikj_loopk:

    lsl w12, w11, #1
    ldr q0, [x1]
    add x1, x1, x12
    ldr q1, [x1]
    add x1, x1, x12
    ldr q2, [x1]
    add x1, x1, x12
    ldr q3, [x1]
    add x1, x1, x12
    ldr q4, [x1]
    add x1, x1, x12
    ldr q5, [x1]
    add x1, x1, x12
    ldr q6, [x1]
    add x1, x1, x12
    ldr q7, [x1]
    add x1, x1, x12
    ldr q8, [x1]
    add x1, x1, x12
    ldr q9, [x1]
    add x1, x1, x12
    ldr q10, [x1]
    add x1, x1, x12
    ldr q11, [x1]
    add x1, x1, x12
    mov w13, #24
    mul w12, w11, w13
    sub x1, x1, x12

// ================

    mov w10, w15
    _ikj_loopj:

    lsl w12, w15, #1
    add x25, x2, x12
    add x26, x25, x12
    add x27, x26, x12
    lsl w13, w12, #2
    ldr q24, [x2]
    ldr q25, [x25]
    ldr q26, [x26]
    ldr q27, [x27]
    ldr q28, [x2, x13]
    ldr q29, [x25, x13]
    ldr q30, [x26, x13]
    ldr q31, [x27, x13]

    add x21, x0, x12
    add x22, x21, x12
    add x23, x22, x12

    ldr q12, [x0]
    ldr q13, [x21]
    ldr q14, [x22]
    ldr q15, [x23]

    mla v12.8H, v24.8H, v0.H[0]
    mla v13.8H, v24.8H, v1.H[0]
    mla v14.8H, v24.8H, v2.H[0]
    mla v15.8H, v24.8H, v3.H[0]

    mla v12.8H, v25.8H, v0.H[1]
    mla v13.8H, v25.8H, v1.H[1]
    mla v14.8H, v25.8H, v2.H[1]
    mla v15.8H, v25.8H, v3.H[1]

    mla v12.8H, v26.8H, v0.H[2]
    mla v13.8H, v26.8H, v1.H[2]
    mla v14.8H, v26.8H, v2.H[2]
    mla v15.8H, v26.8H, v3.H[2]

    mla v12.8H, v27.8H, v0.H[3]
    mla v13.8H, v27.8H, v1.H[3]
    mla v14.8H, v27.8H, v2.H[3]
    mla v15.8H, v27.8H, v3.H[3]

    mla v12.8H, v28.8H, v0.H[4]
    mla v13.8H, v28.8H, v1.H[4]
    mla v14.8H, v28.8H, v2.H[4]
    mla v15.8H, v28.8H, v3.H[4]

    mla v12.8H, v29.8H, v0.H[5]
    mla v13.8H, v29.8H, v1.H[5]
    mla v14.8H, v29.8H, v2.H[5]
    mla v15.8H, v29.8H, v3.H[5]

    mla v12.8H, v30.8H, v0.H[6]
    mla v13.8H, v30.8H, v1.H[6]
    mla v14.8H, v30.8H, v2.H[6]
    mla v15.8H, v30.8H, v3.H[6]

    mla v12.8H, v31.8H, v0.H[7]
    mla v13.8H, v31.8H, v1.H[7]
    mla v14.8H, v31.8H, v2.H[7]
    mla v15.8H, v31.8H, v3.H[7]

    str q12, [x0]
    str q13, [x21]
    str q14, [x22]
    str q15, [x23]

    lsl w12, w15, #3

    ldr q16, [x0, x12]
    ldr q17, [x21, x12]
    ldr q18, [x22, x12]
    ldr q19, [x23, x12]

    mla v16.8H, v24.8H, v4.H[0]
    mla v17.8H, v24.8H, v5.H[0]
    mla v18.8H, v24.8H, v6.H[0]
    mla v19.8H, v24.8H, v7.H[0]

    mla v16.8H, v25.8H, v4.H[1]
    mla v17.8H, v25.8H, v5.H[1]
    mla v18.8H, v25.8H, v6.H[1]
    mla v19.8H, v25.8H, v7.H[1]

    mla v16.8H, v26.8H, v4.H[2]
    mla v17.8H, v26.8H, v5.H[2]
    mla v18.8H, v26.8H, v6.H[2]
    mla v19.8H, v26.8H, v7.H[2]

    mla v16.8H, v27.8H, v4.H[3]
    mla v17.8H, v27.8H, v5.H[3]
    mla v18.8H, v27.8H, v6.H[3]
    mla v19.8H, v27.8H, v7.H[3]

    mla v16.8H, v28.8H, v4.H[4]
    mla v17.8H, v28.8H, v5.H[4]
    mla v18.8H, v28.8H, v6.H[4]
    mla v19.8H, v28.8H, v7.H[4]

    mla v16.8H, v29.8H, v4.H[5]
    mla v17.8H, v29.8H, v5.H[5]
    mla v18.8H, v29.8H, v6.H[5]
    mla v19.8H, v29.8H, v7.H[5]

    mla v16.8H, v30.8H, v4.H[6]
    mla v17.8H, v30.8H, v5.H[6]
    mla v18.8H, v30.8H, v6.H[6]
    mla v19.8H, v30.8H, v7.H[6]

    mla v16.8H, v31.8H, v4.H[7]
    mla v17.8H, v31.8H, v5.H[7]
    mla v18.8H, v31.8H, v6.H[7]
    mla v19.8H, v31.8H, v7.H[7]

    str q16, [x0, x12]
    str q17, [x21, x12]
    str q18, [x22, x12]
    str q19, [x23, x12]

    lsl w12, w15, #4

    ldr q20, [x0, x12]
    ldr q21, [x21, x12]
    ldr q22, [x22, x12]
    ldr q23, [x23, x12]

    mla v20.8H, v24.8H, v8.H[0]
    mla v21.8H, v24.8H, v9.H[0]
    mla v22.8H, v24.8H, v10.H[0]
    mla v23.8H, v24.8H, v11.H[0]

    mla v20.8H, v25.8H, v8.H[1]
    mla v21.8H, v25.8H, v9.H[1]
    mla v22.8H, v25.8H, v10.H[1]
    mla v23.8H, v25.8H, v11.H[1]

    mla v20.8H, v26.8H, v8.H[2]
    mla v21.8H, v26.8H, v9.H[2]
    mla v22.8H, v26.8H, v10.H[2]
    mla v23.8H, v26.8H, v11.H[2]

    mla v20.8H, v27.8H, v8.H[3]
    mla v21.8H, v27.8H, v9.H[3]
    mla v22.8H, v27.8H, v10.H[3]
    mla v23.8H, v27.8H, v11.H[3]

    mla v20.8H, v28.8H, v8.H[4]
    mla v21.8H, v28.8H, v9.H[4]
    mla v22.8H, v28.8H, v10.H[4]
    mla v23.8H, v28.8H, v11.H[4]

    mla v20.8H, v29.8H, v8.H[5]
    mla v21.8H, v29.8H, v9.H[5]
    mla v22.8H, v29.8H, v10.H[5]
    mla v23.8H, v29.8H, v11.H[5]

    mla v20.8H, v30.8H, v8.H[6]
    mla v21.8H, v30.8H, v9.H[6]
    mla v22.8H, v30.8H, v10.H[6]
    mla v23.8H, v30.8H, v11.H[6]

    mla v20.8H, v31.8H, v8.H[7]
    mla v21.8H, v31.8H, v9.H[7]
    mla v22.8H, v31.8H, v10.H[7]
    mla v23.8H, v31.8H, v11.H[7]

    str q20, [x0, x12]
    str q21, [x21, x12]
    str q22, [x22, x12]
    str q23, [x23, x12]

    add x0, x0, #8*2
    add x2, x2, #8*2

    sub w10, w10, #8
    cbnz w10, _ikj_loopj

    lsl w12, w15, #1
    sub x0, x0, x12
    sub x2, x2, x12

// ================

    add x1, x1, #8*2
    lsl w12, w15, #4
    add x2, x2, x12

    sub w9, w9, #8
    cbnz w9, _ikj_loopk

    lsl w12, w11, #1
    sub x1, x1, x12
    mul w13, w12, w15
    sub x2, x2, x13

// ================

    mov w14, #24
    mul w12, w15, w14
    add x0, x0, x12
    mul w13, w11, w14
    add x1, x1, x13

    sub w7, w7, #12
    cbnz w7, _ikj_loopi

// ================

    pop_all

    br lr


.align 8
.global ikj_matmul_addsrc1_asm
.global _ikj_matmul_addsrc1_asm
ikj_matmul_addsrc1_asm:
_ikj_matmul_addsrc1_asm:

    push_all

//    for(size_t i = 0; i < DIM_I; i++){
//        for(size_t k = 0; k < DIM_K; k++){
//            for(size_t j = 0; j < DIM_J; j++){
//                des[i][j] += src1[i][k] * src2[k][j];
//            }
//        }
//    }

    ldr w7, [x3, #0]
    ldr w11, [x3, #4]
    ldr w15, [x3, #8]

// ================

    _ikj_addsrc1_loopi:

// ================

    mov w9, w11
    _ikj_addsrc1_loopk:

    lsl w12, w11, #1
    ldr q0, [x1]
    ldr q12, [x4]
    add v0.8H, v0.8H, v12.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q1, [x1]
    ldr q13, [x4]
    add v1.8H, v1.8H, v13.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q2, [x1]
    ldr q14, [x4]
    add v2.8H, v2.8H, v14.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q3, [x1]
    ldr q15, [x4]
    add v3.8H, v3.8H, v15.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q4, [x1]
    ldr q12, [x4]
    add v4.8H, v4.8H, v12.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q5, [x1]
    ldr q13, [x4]
    add v5.8H, v5.8H, v13.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q6, [x1]
    ldr q14, [x4]
    add v6.8H, v6.8H, v14.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q7, [x1]
    ldr q15, [x4]
    add v7.8H, v7.8H, v15.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q8, [x1]
    ldr q12, [x4]
    add v8.8H, v8.8H, v12.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q9, [x1]
    ldr q13, [x4]
    add v9.8H, v9.8H, v13.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q10, [x1]
    ldr q14, [x4]
    add v10.8H, v10.8H, v14.8H
    add x1, x1, x12
    add x4, x4, x12
    ldr q11, [x1]
    ldr q15, [x4]
    add v11.8H, v11.8H, v15.8H
    add x1, x1, x12
    add x4, x4, x12
    mov w13, #24
    mul w12, w11, w13
    sub x1, x1, x12
    sub x4, x4, x12

// ================

    mov w10, w15
    _ikj_addsrc1_loopj:

    lsl w12, w15, #1
    add x25, x2, x12
    add x26, x25, x12
    add x27, x26, x12
    lsl w13, w12, #2
    ldr q24, [x2]
    ldr q25, [x25]
    ldr q26, [x26]
    ldr q27, [x27]
    ldr q28, [x2, x13]
    ldr q29, [x25, x13]
    ldr q30, [x26, x13]
    ldr q31, [x27, x13]

    add x21, x0, x12
    add x22, x21, x12
    add x23, x22, x12

    ldr q12, [x0]
    ldr q13, [x21]
    ldr q14, [x22]
    ldr q15, [x23]

    mla v12.8H, v24.8H, v0.H[0]
    mla v13.8H, v24.8H, v1.H[0]
    mla v14.8H, v24.8H, v2.H[0]
    mla v15.8H, v24.8H, v3.H[0]

    mla v12.8H, v25.8H, v0.H[1]
    mla v13.8H, v25.8H, v1.H[1]
    mla v14.8H, v25.8H, v2.H[1]
    mla v15.8H, v25.8H, v3.H[1]

    mla v12.8H, v26.8H, v0.H[2]
    mla v13.8H, v26.8H, v1.H[2]
    mla v14.8H, v26.8H, v2.H[2]
    mla v15.8H, v26.8H, v3.H[2]

    mla v12.8H, v27.8H, v0.H[3]
    mla v13.8H, v27.8H, v1.H[3]
    mla v14.8H, v27.8H, v2.H[3]
    mla v15.8H, v27.8H, v3.H[3]

    mla v12.8H, v28.8H, v0.H[4]
    mla v13.8H, v28.8H, v1.H[4]
    mla v14.8H, v28.8H, v2.H[4]
    mla v15.8H, v28.8H, v3.H[4]

    mla v12.8H, v29.8H, v0.H[5]
    mla v13.8H, v29.8H, v1.H[5]
    mla v14.8H, v29.8H, v2.H[5]
    mla v15.8H, v29.8H, v3.H[5]

    mla v12.8H, v30.8H, v0.H[6]
    mla v13.8H, v30.8H, v1.H[6]
    mla v14.8H, v30.8H, v2.H[6]
    mla v15.8H, v30.8H, v3.H[6]

    mla v12.8H, v31.8H, v0.H[7]
    mla v13.8H, v31.8H, v1.H[7]
    mla v14.8H, v31.8H, v2.H[7]
    mla v15.8H, v31.8H, v3.H[7]

    str q12, [x0]
    str q13, [x21]
    str q14, [x22]
    str q15, [x23]

    lsl w12, w15, #3

    ldr q16, [x0, x12]
    ldr q17, [x21, x12]
    ldr q18, [x22, x12]
    ldr q19, [x23, x12]

    mla v16.8H, v24.8H, v4.H[0]
    mla v17.8H, v24.8H, v5.H[0]
    mla v18.8H, v24.8H, v6.H[0]
    mla v19.8H, v24.8H, v7.H[0]

    mla v16.8H, v25.8H, v4.H[1]
    mla v17.8H, v25.8H, v5.H[1]
    mla v18.8H, v25.8H, v6.H[1]
    mla v19.8H, v25.8H, v7.H[1]

    mla v16.8H, v26.8H, v4.H[2]
    mla v17.8H, v26.8H, v5.H[2]
    mla v18.8H, v26.8H, v6.H[2]
    mla v19.8H, v26.8H, v7.H[2]

    mla v16.8H, v27.8H, v4.H[3]
    mla v17.8H, v27.8H, v5.H[3]
    mla v18.8H, v27.8H, v6.H[3]
    mla v19.8H, v27.8H, v7.H[3]

    mla v16.8H, v28.8H, v4.H[4]
    mla v17.8H, v28.8H, v5.H[4]
    mla v18.8H, v28.8H, v6.H[4]
    mla v19.8H, v28.8H, v7.H[4]

    mla v16.8H, v29.8H, v4.H[5]
    mla v17.8H, v29.8H, v5.H[5]
    mla v18.8H, v29.8H, v6.H[5]
    mla v19.8H, v29.8H, v7.H[5]

    mla v16.8H, v30.8H, v4.H[6]
    mla v17.8H, v30.8H, v5.H[6]
    mla v18.8H, v30.8H, v6.H[6]
    mla v19.8H, v30.8H, v7.H[6]

    mla v16.8H, v31.8H, v4.H[7]
    mla v17.8H, v31.8H, v5.H[7]
    mla v18.8H, v31.8H, v6.H[7]
    mla v19.8H, v31.8H, v7.H[7]

    str q16, [x0, x12]
    str q17, [x21, x12]
    str q18, [x22, x12]
    str q19, [x23, x12]

    lsl w12, w15, #4

    ldr q20, [x0, x12]
    ldr q21, [x21, x12]
    ldr q22, [x22, x12]
    ldr q23, [x23, x12]

    mla v20.8H, v24.8H, v8.H[0]
    mla v21.8H, v24.8H, v9.H[0]
    mla v22.8H, v24.8H, v10.H[0]
    mla v23.8H, v24.8H, v11.H[0]

    mla v20.8H, v25.8H, v8.H[1]
    mla v21.8H, v25.8H, v9.H[1]
    mla v22.8H, v25.8H, v10.H[1]
    mla v23.8H, v25.8H, v11.H[1]

    mla v20.8H, v26.8H, v8.H[2]
    mla v21.8H, v26.8H, v9.H[2]
    mla v22.8H, v26.8H, v10.H[2]
    mla v23.8H, v26.8H, v11.H[2]

    mla v20.8H, v27.8H, v8.H[3]
    mla v21.8H, v27.8H, v9.H[3]
    mla v22.8H, v27.8H, v10.H[3]
    mla v23.8H, v27.8H, v11.H[3]

    mla v20.8H, v28.8H, v8.H[4]
    mla v21.8H, v28.8H, v9.H[4]
    mla v22.8H, v28.8H, v10.H[4]
    mla v23.8H, v28.8H, v11.H[4]

    mla v20.8H, v29.8H, v8.H[5]
    mla v21.8H, v29.8H, v9.H[5]
    mla v22.8H, v29.8H, v10.H[5]
    mla v23.8H, v29.8H, v11.H[5]

    mla v20.8H, v30.8H, v8.H[6]
    mla v21.8H, v30.8H, v9.H[6]
    mla v22.8H, v30.8H, v10.H[6]
    mla v23.8H, v30.8H, v11.H[6]

    mla v20.8H, v31.8H, v8.H[7]
    mla v21.8H, v31.8H, v9.H[7]
    mla v22.8H, v31.8H, v10.H[7]
    mla v23.8H, v31.8H, v11.H[7]

    str q20, [x0, x12]
    str q21, [x21, x12]
    str q22, [x22, x12]
    str q23, [x23, x12]

    add x0, x0, #8*2
    add x2, x2, #8*2

    sub w10, w10, #8
    cbnz w10, _ikj_addsrc1_loopj

    lsl w12, w15, #1
    sub x0, x0, x12
    sub x2, x2, x12

// ================

    add x1, x1, #8*2
    add x4, x4, #8*2
    lsl w12, w15, #4
    add x2, x2, x12

    sub w9, w9, #8
    cbnz w9, _ikj_addsrc1_loopk

    lsl w12, w11, #1
    sub x1, x1, x12
    sub x4, x4, x12
    mul w13, w12, w15
    sub x2, x2, x13

// ================

    mov w14, #24
    mul w12, w15, w14
    add x0, x0, x12
    mul w13, w11, w14
    add x1, x1, x13
    add x4, x4, x13

    sub w7, w7, #12
    cbnz w7, _ikj_addsrc1_loopi

// ================

    pop_all

    br lr


.align 8
.global ikj_matmul_subnegaccsrc1_asm
.global _ikj_matmul_subnegaccsrc1_asm
ikj_matmul_subnegaccsrc1_asm:
_ikj_matmul_subnegaccsrc1_asm:

    push_all

//    for(size_t i = 0; i < DIM_I; i++){
//        for(size_t k = 0; k < DIM_K; k++){
//            for(size_t j = 0; j < DIM_J; j++){
//                des[i][j] += src1[i][k] * src2[k][j];
//            }
//        }
//    }

    ldr w7, [x3, #0]
    ldr w11, [x3, #4]
    ldr w15, [x3, #8]

// ================

    _ikj_subnegaccsrc1_loopi:

// ================

    mov w9, w11
    _ikj_subnegaccsrc1_loopk:

    lsl w12, w11, #1
    ldr q0, [x1]
    ldr q12, [x4]
    ldr q16, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q1, [x1]
    ldr q13, [x4]
    ldr q17, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q2, [x1]
    ldr q14, [x4]
    ldr q18, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q3, [x1]
    ldr q15, [x4]
    ldr q19, [x5]

    add  v0.8H,  v0.8H, v16.8H
    sub  v0.8H, v12.8H,  v0.8H
    add  v1.8H,  v1.8H, v17.8H
    sub  v1.8H, v13.8H,  v1.8H
    add  v2.8H,  v2.8H, v18.8H
    sub  v2.8H, v14.8H,  v2.8H
    add  v3.8H,  v3.8H, v19.8H
    sub  v3.8H, v15.8H,  v3.8H

    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q4, [x1]
    ldr q20, [x4]
    ldr q24, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q5, [x1]
    ldr q21, [x4]
    ldr q25, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q6, [x1]
    ldr q22, [x4]
    ldr q26, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q7, [x1]
    ldr q23, [x4]
    ldr q27, [x5]

    add  v4.8H,  v4.8H, v24.8H
    sub  v4.8H, v20.8H,  v4.8H
    add  v5.8H,  v5.8H, v25.8H
    sub  v5.8H, v21.8H,  v5.8H
    add  v6.8H,  v6.8H, v26.8H
    sub  v6.8H, v22.8H,  v6.8H
    add  v7.8H,  v7.8H, v27.8H
    sub  v7.8H, v23.8H,  v7.8H

    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q8, [x1]
    ldr q12, [x4]
    ldr q16, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q9, [x1]
    ldr q13, [x4]
    ldr q17, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q10, [x1]
    ldr q14, [x4]
    ldr q18, [x5]
    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    ldr q11, [x1]
    ldr q15, [x4]
    ldr q19, [x5]

    add  v8.8H,  v8.8H, v16.8H
    sub  v8.8H, v12.8H,  v8.8H
    add  v9.8H,  v9.8H, v17.8H
    sub  v9.8H, v13.8H,  v9.8H
    add v10.8H, v10.8H, v18.8H
    sub v10.8H, v14.8H, v10.8H
    add v11.8H, v11.8H, v19.8H
    sub v11.8H, v15.8H, v11.8H

    add x1, x1, x12
    add x4, x4, x12
    add x5, x5, x12
    mov w13, #24
    mul w12, w11, w13
    sub x1, x1, x12
    sub x4, x4, x12
    sub x5, x5, x12

// ================

    mov w10, w15
    _ikj_subnegaccsrc1_loopj:

    lsl w12, w15, #1
    add x25, x2, x12
    add x26, x25, x12
    add x27, x26, x12
    lsl w13, w12, #2
    ldr q24, [x2]
    ldr q25, [x25]
    ldr q26, [x26]
    ldr q27, [x27]
    ldr q28, [x2, x13]
    ldr q29, [x25, x13]
    ldr q30, [x26, x13]
    ldr q31, [x27, x13]

    add x21, x0, x12
    add x22, x21, x12
    add x23, x22, x12

    ldr q12, [x0]
    ldr q13, [x21]
    ldr q14, [x22]
    ldr q15, [x23]

    mla v12.8H, v24.8H, v0.H[0]
    mla v13.8H, v24.8H, v1.H[0]
    mla v14.8H, v24.8H, v2.H[0]
    mla v15.8H, v24.8H, v3.H[0]

    mla v12.8H, v25.8H, v0.H[1]
    mla v13.8H, v25.8H, v1.H[1]
    mla v14.8H, v25.8H, v2.H[1]
    mla v15.8H, v25.8H, v3.H[1]

    mla v12.8H, v26.8H, v0.H[2]
    mla v13.8H, v26.8H, v1.H[2]
    mla v14.8H, v26.8H, v2.H[2]
    mla v15.8H, v26.8H, v3.H[2]

    mla v12.8H, v27.8H, v0.H[3]
    mla v13.8H, v27.8H, v1.H[3]
    mla v14.8H, v27.8H, v2.H[3]
    mla v15.8H, v27.8H, v3.H[3]

    mla v12.8H, v28.8H, v0.H[4]
    mla v13.8H, v28.8H, v1.H[4]
    mla v14.8H, v28.8H, v2.H[4]
    mla v15.8H, v28.8H, v3.H[4]

    mla v12.8H, v29.8H, v0.H[5]
    mla v13.8H, v29.8H, v1.H[5]
    mla v14.8H, v29.8H, v2.H[5]
    mla v15.8H, v29.8H, v3.H[5]

    mla v12.8H, v30.8H, v0.H[6]
    mla v13.8H, v30.8H, v1.H[6]
    mla v14.8H, v30.8H, v2.H[6]
    mla v15.8H, v30.8H, v3.H[6]

    mla v12.8H, v31.8H, v0.H[7]
    mla v13.8H, v31.8H, v1.H[7]
    mla v14.8H, v31.8H, v2.H[7]
    mla v15.8H, v31.8H, v3.H[7]

    str q12, [x0]
    str q13, [x21]
    str q14, [x22]
    str q15, [x23]

    lsl w12, w15, #3

    ldr q16, [x0, x12]
    ldr q17, [x21, x12]
    ldr q18, [x22, x12]
    ldr q19, [x23, x12]

    mla v16.8H, v24.8H, v4.H[0]
    mla v17.8H, v24.8H, v5.H[0]
    mla v18.8H, v24.8H, v6.H[0]
    mla v19.8H, v24.8H, v7.H[0]

    mla v16.8H, v25.8H, v4.H[1]
    mla v17.8H, v25.8H, v5.H[1]
    mla v18.8H, v25.8H, v6.H[1]
    mla v19.8H, v25.8H, v7.H[1]

    mla v16.8H, v26.8H, v4.H[2]
    mla v17.8H, v26.8H, v5.H[2]
    mla v18.8H, v26.8H, v6.H[2]
    mla v19.8H, v26.8H, v7.H[2]

    mla v16.8H, v27.8H, v4.H[3]
    mla v17.8H, v27.8H, v5.H[3]
    mla v18.8H, v27.8H, v6.H[3]
    mla v19.8H, v27.8H, v7.H[3]

    mla v16.8H, v28.8H, v4.H[4]
    mla v17.8H, v28.8H, v5.H[4]
    mla v18.8H, v28.8H, v6.H[4]
    mla v19.8H, v28.8H, v7.H[4]

    mla v16.8H, v29.8H, v4.H[5]
    mla v17.8H, v29.8H, v5.H[5]
    mla v18.8H, v29.8H, v6.H[5]
    mla v19.8H, v29.8H, v7.H[5]

    mla v16.8H, v30.8H, v4.H[6]
    mla v17.8H, v30.8H, v5.H[6]
    mla v18.8H, v30.8H, v6.H[6]
    mla v19.8H, v30.8H, v7.H[6]

    mla v16.8H, v31.8H, v4.H[7]
    mla v17.8H, v31.8H, v5.H[7]
    mla v18.8H, v31.8H, v6.H[7]
    mla v19.8H, v31.8H, v7.H[7]

    str q16, [x0, x12]
    str q17, [x21, x12]
    str q18, [x22, x12]
    str q19, [x23, x12]

    lsl w12, w15, #4

    ldr q20, [x0, x12]
    ldr q21, [x21, x12]
    ldr q22, [x22, x12]
    ldr q23, [x23, x12]

    mla v20.8H, v24.8H, v8.H[0]
    mla v21.8H, v24.8H, v9.H[0]
    mla v22.8H, v24.8H, v10.H[0]
    mla v23.8H, v24.8H, v11.H[0]

    mla v20.8H, v25.8H, v8.H[1]
    mla v21.8H, v25.8H, v9.H[1]
    mla v22.8H, v25.8H, v10.H[1]
    mla v23.8H, v25.8H, v11.H[1]

    mla v20.8H, v26.8H, v8.H[2]
    mla v21.8H, v26.8H, v9.H[2]
    mla v22.8H, v26.8H, v10.H[2]
    mla v23.8H, v26.8H, v11.H[2]

    mla v20.8H, v27.8H, v8.H[3]
    mla v21.8H, v27.8H, v9.H[3]
    mla v22.8H, v27.8H, v10.H[3]
    mla v23.8H, v27.8H, v11.H[3]

    mla v20.8H, v28.8H, v8.H[4]
    mla v21.8H, v28.8H, v9.H[4]
    mla v22.8H, v28.8H, v10.H[4]
    mla v23.8H, v28.8H, v11.H[4]

    mla v20.8H, v29.8H, v8.H[5]
    mla v21.8H, v29.8H, v9.H[5]
    mla v22.8H, v29.8H, v10.H[5]
    mla v23.8H, v29.8H, v11.H[5]

    mla v20.8H, v30.8H, v8.H[6]
    mla v21.8H, v30.8H, v9.H[6]
    mla v22.8H, v30.8H, v10.H[6]
    mla v23.8H, v30.8H, v11.H[6]

    mla v20.8H, v31.8H, v8.H[7]
    mla v21.8H, v31.8H, v9.H[7]
    mla v22.8H, v31.8H, v10.H[7]
    mla v23.8H, v31.8H, v11.H[7]

    str q20, [x0, x12]
    str q21, [x21, x12]
    str q22, [x22, x12]
    str q23, [x23, x12]

    add x0, x0, #8*2
    add x2, x2, #8*2

    sub w10, w10, #8
    cbnz w10, _ikj_subnegaccsrc1_loopj

    lsl w12, w15, #1
    sub x0, x0, x12
    sub x2, x2, x12

// ================

    add x1, x1, #8*2
    add x4, x4, #8*2
    add x5, x5, #8*2
    lsl w12, w15, #4
    add x2, x2, x12

    sub w9, w9, #8
    cbnz w9, _ikj_subnegaccsrc1_loopk

    lsl w12, w11, #1
    sub x1, x1, x12
    sub x4, x4, x12
    sub x5, x5, x12
    mul w13, w12, w15
    sub x2, x2, x13

// ================

    mov w14, #24
    mul w12, w15, w14
    add x0, x0, x12
    mul w13, w11, w14
    add x1, x1, x13
    add x4, x4, x13
    add x5, x5, x13

    sub w7, w7, #12
    cbnz w7, _ikj_subnegaccsrc1_loopi

// ================

    pop_all

    br lr


.align 8
.global ijk_matmul_asm
.global _ijk_matmul_asm
ijk_matmul_asm:
_ijk_matmul_asm:

    push_all

//    for(size_t i = 0; i < DIM_I; i++){
//        for(size_t j = 0; j < DIM_J; j++){
//            des[i][j] = 0;
//            for(size_t k = 0; k < DIM_K; k++){
//                des[i][j] += src1[i][k] * src2[k][j];
//            }
//        }
//    }

    ldr w7, [x3, #0]
    ldr w11, [x3, #4]
    ldr w15, [x3, #8]


// ================

    _ijk_loopi:

// ================

    mov w9, w15
    _ijk_loopj:

    mov w12, #0
    dup v20.8H, w12
    dup v21.8H, w12
    dup v22.8H, w12
    dup v23.8H, w12
    dup v24.8H, w12
    dup v25.8H, w12
    dup v26.8H, w12
    dup v27.8H, w12
    dup v28.8H, w12
    dup v29.8H, w12
    dup v30.8H, w12
    dup v31.8H, w12

// ================

    mov w10, w11
    _ijk_loopk:

    lsl w12, w11, #1

    add x21, x1, x12
    add x22, x21, x12
    add x23, x22, x12

    ldr q0, [x1]
    ldr q1, [x21]
    ldr q2, [x22]
    ldr q3, [x23]

    lsl w12, w11, #3
    ldr q4, [x1, x12]
    ldr q5, [x21, x12]
    ldr q6, [x22, x12]
    ldr q7, [x23, x12]

    lsl w12, w11, #4
    ldr q8, [x1, x12]
    ldr q9, [x21, x12]
    ldr q10, [x22, x12]
    ldr q11, [x23, x12]

    lsl w12, w11, #1
    add x25, x2, x12
    add x26, x25, x12
    add x27, x26, x12
    ldr q12, [x2]
    ldr q13, [x25]
    ldr q14, [x26]
    ldr q15, [x27]

    mla v20.8H, v12.8H, v0.H[0]
    mla v21.8H, v12.8H, v1.H[0]
    mla v22.8H, v12.8H, v2.H[0]
    mla v23.8H, v12.8H, v3.H[0]
    mla v24.8H, v12.8H, v4.H[0]
    mla v25.8H, v12.8H, v5.H[0]
    mla v26.8H, v12.8H, v6.H[0]
    mla v27.8H, v12.8H, v7.H[0]
    mla v28.8H, v12.8H, v8.H[0]
    mla v29.8H, v12.8H, v9.H[0]
    mla v30.8H, v12.8H, v10.H[0]
    mla v31.8H, v12.8H, v11.H[0]

    mla v20.8H, v13.8H, v0.H[1]
    mla v21.8H, v13.8H, v1.H[1]
    mla v22.8H, v13.8H, v2.H[1]
    mla v23.8H, v13.8H, v3.H[1]
    mla v24.8H, v13.8H, v4.H[1]
    mla v25.8H, v13.8H, v5.H[1]
    mla v26.8H, v13.8H, v6.H[1]
    mla v27.8H, v13.8H, v7.H[1]
    mla v28.8H, v13.8H, v8.H[1]
    mla v29.8H, v13.8H, v9.H[1]
    mla v30.8H, v13.8H, v10.H[1]
    mla v31.8H, v13.8H, v11.H[1]

    lsl w12, w11, #3
    ldr q16, [x2, x12]
    ldr q17, [x25, x12]
    ldr q18, [x26, x12]
    ldr q19, [x27, x12]

    mla v20.8H, v14.8H, v0.H[2]
    mla v21.8H, v14.8H, v1.H[2]
    mla v22.8H, v14.8H, v2.H[2]
    mla v23.8H, v14.8H, v3.H[2]
    mla v24.8H, v14.8H, v4.H[2]
    mla v25.8H, v14.8H, v5.H[2]
    mla v26.8H, v14.8H, v6.H[2]
    mla v27.8H, v14.8H, v7.H[2]
    mla v28.8H, v14.8H, v8.H[2]
    mla v29.8H, v14.8H, v9.H[2]
    mla v30.8H, v14.8H, v10.H[2]
    mla v31.8H, v14.8H, v11.H[2]

    mla v20.8H, v15.8H, v0.H[3]
    mla v21.8H, v15.8H, v1.H[3]
    mla v22.8H, v15.8H, v2.H[3]
    mla v23.8H, v15.8H, v3.H[3]
    mla v24.8H, v15.8H, v4.H[3]
    mla v25.8H, v15.8H, v5.H[3]
    mla v26.8H, v15.8H, v6.H[3]
    mla v27.8H, v15.8H, v7.H[3]
    mla v28.8H, v15.8H, v8.H[3]
    mla v29.8H, v15.8H, v9.H[3]
    mla v30.8H, v15.8H, v10.H[3]
    mla v31.8H, v15.8H, v11.H[3]

    mla v20.8H, v16.8H, v0.H[4]
    mla v21.8H, v16.8H, v1.H[4]
    mla v22.8H, v16.8H, v2.H[4]
    mla v23.8H, v16.8H, v3.H[4]
    mla v24.8H, v16.8H, v4.H[4]
    mla v25.8H, v16.8H, v5.H[4]
    mla v26.8H, v16.8H, v6.H[4]
    mla v27.8H, v16.8H, v7.H[4]
    mla v28.8H, v16.8H, v8.H[4]
    mla v29.8H, v16.8H, v9.H[4]
    mla v30.8H, v16.8H, v10.H[4]
    mla v31.8H, v16.8H, v11.H[4]

    mla v20.8H, v17.8H, v0.H[5]
    mla v21.8H, v17.8H, v1.H[5]
    mla v22.8H, v17.8H, v2.H[5]
    mla v23.8H, v17.8H, v3.H[5]
    mla v24.8H, v17.8H, v4.H[5]
    mla v25.8H, v17.8H, v5.H[5]
    mla v26.8H, v17.8H, v6.H[5]
    mla v27.8H, v17.8H, v7.H[5]
    mla v28.8H, v17.8H, v8.H[5]
    mla v29.8H, v17.8H, v9.H[5]
    mla v30.8H, v17.8H, v10.H[5]
    mla v31.8H, v17.8H, v11.H[5]

    mla v20.8H, v18.8H, v0.H[6]
    mla v21.8H, v18.8H, v1.H[6]
    mla v22.8H, v18.8H, v2.H[6]
    mla v23.8H, v18.8H, v3.H[6]
    mla v24.8H, v18.8H, v4.H[6]
    mla v25.8H, v18.8H, v5.H[6]
    mla v26.8H, v18.8H, v6.H[6]
    mla v27.8H, v18.8H, v7.H[6]
    mla v28.8H, v18.8H, v8.H[6]
    mla v29.8H, v18.8H, v9.H[6]
    mla v30.8H, v18.8H, v10.H[6]
    mla v31.8H, v18.8H, v11.H[6]

    mla v20.8H, v19.8H, v0.H[7]
    mla v21.8H, v19.8H, v1.H[7]
    mla v22.8H, v19.8H, v2.H[7]
    mla v23.8H, v19.8H, v3.H[7]
    mla v24.8H, v19.8H, v4.H[7]
    mla v25.8H, v19.8H, v5.H[7]
    mla v26.8H, v19.8H, v6.H[7]
    mla v27.8H, v19.8H, v7.H[7]
    mla v28.8H, v19.8H, v8.H[7]
    mla v29.8H, v19.8H, v9.H[7]
    mla v30.8H, v19.8H, v10.H[7]
    mla v31.8H, v19.8H, v11.H[7]

    add x1, x1, #16
    lsl w12, w15, #4
    add x2, x2, x12

    sub w10, w10, #8
    cbnz w10, _ijk_loopk

// ================

    lsl w12, w15, #1
    add x25, x0, x12
    add x26, x25, x12
    add x27, x26, x12

    str q20, [x0]
    str q21, [x25]
    str q22, [x26]
    str q23, [x27]

    lsl w12, w15, #3
    str q24, [x0, x12]
    str q25, [x25, x12]
    str q26, [x26, x12]
    str q27, [x27, x12]

    lsl w12, w15, #4
    str q28, [x0, x12]
    str q29, [x25, x12]
    str q30, [x26, x12]
    str q31, [x27, x12]

    lsl w12, w11, #1
    sub x1, x1, x12

    mul w12, w15, w11
    lsl w12, w12, #1
    sub x2, x2, x12

    add x0, x0, #16
    add x2, x2, #16

    sub w9, w9, #8
    cbnz w9, _ijk_loopj

// ================

    lsl w12, w15, #1
    sub x2, x2, x12

    lsl w12, w15, #1
    sub x0, x0, x12
    mov w13, #24
    mul w12, w15, w13
    add x0, x0, x12
    add x1, x1, x12

    sub w7, w7, #12
    cbnz w7, _ijk_loopi

// ================

    pop_all

    br lr


.align 8
.global ijk_matmla_asm
.global _ijk_matmla_asm
ijk_matmla_asm:
_ijk_matmla_asm:

    push_all

//    for(size_t i = 0; i < DIM_I; i++){
//        for(size_t j = 0; j < DIM_J; j++){
//            for(size_t k = 0; k < DIM_K; k++){
//                des[i][j] += src1[i][k] * src2[k][j];
//            }
//        }
//    }

    ldr w7, [x3, #0]
    ldr w11, [x3, #4]
    ldr w15, [x3, #8]

// ================

    _ijk_mla_loopi:

// ================

    mov w9, w15
    _ijk_mla_loopj:

    lsl w12, w15, #1
    add x25, x0, x12
    add x26, x25, x12
    add x27, x26, x12

    ldr q20, [x0]
    ldr q21, [x25]
    ldr q22, [x26]
    ldr q23, [x27]

    lsl w12, w15, #3
    ldr q24, [x0, x12]
    ldr q25, [x25, x12]
    ldr q26, [x26, x12]
    ldr q27, [x27, x12]

    lsl w12, w15, #4
    ldr q28, [x0, x12]
    ldr q29, [x25, x12]
    ldr q30, [x26, x12]
    ldr q31, [x27, x12]

// ================

    mov w10, w11
    _ijk_mla_loopk:

    lsl w12, w11, #1

    add x21, x1, x12
    add x22, x21, x12
    add x23, x22, x12

    ldr q0, [x1]
    ldr q1, [x21]
    ldr q2, [x22]
    ldr q3, [x23]

    lsl w12, w11, #3
    ldr q4, [x1, x12]
    ldr q5, [x21, x12]
    ldr q6, [x22, x12]
    ldr q7, [x23, x12]

    lsl w12, w11, #4
    ldr q8, [x1, x12]
    ldr q9, [x21, x12]
    ldr q10, [x22, x12]
    ldr q11, [x23, x12]

    lsl w12, w11, #1
    add x25, x2, x12
    add x26, x25, x12
    add x27, x26, x12
    ldr q12, [x2]
    ldr q13, [x25]
    ldr q14, [x26]
    ldr q15, [x27]

    mla v20.8H, v12.8H, v0.H[0]
    mla v21.8H, v12.8H, v1.H[0]
    mla v22.8H, v12.8H, v2.H[0]
    mla v23.8H, v12.8H, v3.H[0]
    mla v24.8H, v12.8H, v4.H[0]
    mla v25.8H, v12.8H, v5.H[0]
    mla v26.8H, v12.8H, v6.H[0]
    mla v27.8H, v12.8H, v7.H[0]
    mla v28.8H, v12.8H, v8.H[0]
    mla v29.8H, v12.8H, v9.H[0]
    mla v30.8H, v12.8H, v10.H[0]
    mla v31.8H, v12.8H, v11.H[0]

    mla v20.8H, v13.8H, v0.H[1]
    mla v21.8H, v13.8H, v1.H[1]
    mla v22.8H, v13.8H, v2.H[1]
    mla v23.8H, v13.8H, v3.H[1]
    mla v24.8H, v13.8H, v4.H[1]
    mla v25.8H, v13.8H, v5.H[1]
    mla v26.8H, v13.8H, v6.H[1]
    mla v27.8H, v13.8H, v7.H[1]
    mla v28.8H, v13.8H, v8.H[1]
    mla v29.8H, v13.8H, v9.H[1]
    mla v30.8H, v13.8H, v10.H[1]
    mla v31.8H, v13.8H, v11.H[1]

    lsl w12, w11, #3
    ldr q16, [x2, x12]
    ldr q17, [x25, x12]
    ldr q18, [x26, x12]
    ldr q19, [x27, x12]

    mla v20.8H, v14.8H, v0.H[2]
    mla v21.8H, v14.8H, v1.H[2]
    mla v22.8H, v14.8H, v2.H[2]
    mla v23.8H, v14.8H, v3.H[2]
    mla v24.8H, v14.8H, v4.H[2]
    mla v25.8H, v14.8H, v5.H[2]
    mla v26.8H, v14.8H, v6.H[2]
    mla v27.8H, v14.8H, v7.H[2]
    mla v28.8H, v14.8H, v8.H[2]
    mla v29.8H, v14.8H, v9.H[2]
    mla v30.8H, v14.8H, v10.H[2]
    mla v31.8H, v14.8H, v11.H[2]

    mla v20.8H, v15.8H, v0.H[3]
    mla v21.8H, v15.8H, v1.H[3]
    mla v22.8H, v15.8H, v2.H[3]
    mla v23.8H, v15.8H, v3.H[3]
    mla v24.8H, v15.8H, v4.H[3]
    mla v25.8H, v15.8H, v5.H[3]
    mla v26.8H, v15.8H, v6.H[3]
    mla v27.8H, v15.8H, v7.H[3]
    mla v28.8H, v15.8H, v8.H[3]
    mla v29.8H, v15.8H, v9.H[3]
    mla v30.8H, v15.8H, v10.H[3]
    mla v31.8H, v15.8H, v11.H[3]

    mla v20.8H, v16.8H, v0.H[4]
    mla v21.8H, v16.8H, v1.H[4]
    mla v22.8H, v16.8H, v2.H[4]
    mla v23.8H, v16.8H, v3.H[4]
    mla v24.8H, v16.8H, v4.H[4]
    mla v25.8H, v16.8H, v5.H[4]
    mla v26.8H, v16.8H, v6.H[4]
    mla v27.8H, v16.8H, v7.H[4]
    mla v28.8H, v16.8H, v8.H[4]
    mla v29.8H, v16.8H, v9.H[4]
    mla v30.8H, v16.8H, v10.H[4]
    mla v31.8H, v16.8H, v11.H[4]

    mla v20.8H, v17.8H, v0.H[5]
    mla v21.8H, v17.8H, v1.H[5]
    mla v22.8H, v17.8H, v2.H[5]
    mla v23.8H, v17.8H, v3.H[5]
    mla v24.8H, v17.8H, v4.H[5]
    mla v25.8H, v17.8H, v5.H[5]
    mla v26.8H, v17.8H, v6.H[5]
    mla v27.8H, v17.8H, v7.H[5]
    mla v28.8H, v17.8H, v8.H[5]
    mla v29.8H, v17.8H, v9.H[5]
    mla v30.8H, v17.8H, v10.H[5]
    mla v31.8H, v17.8H, v11.H[5]

    mla v20.8H, v18.8H, v0.H[6]
    mla v21.8H, v18.8H, v1.H[6]
    mla v22.8H, v18.8H, v2.H[6]
    mla v23.8H, v18.8H, v3.H[6]
    mla v24.8H, v18.8H, v4.H[6]
    mla v25.8H, v18.8H, v5.H[6]
    mla v26.8H, v18.8H, v6.H[6]
    mla v27.8H, v18.8H, v7.H[6]
    mla v28.8H, v18.8H, v8.H[6]
    mla v29.8H, v18.8H, v9.H[6]
    mla v30.8H, v18.8H, v10.H[6]
    mla v31.8H, v18.8H, v11.H[6]

    mla v20.8H, v19.8H, v0.H[7]
    mla v21.8H, v19.8H, v1.H[7]
    mla v22.8H, v19.8H, v2.H[7]
    mla v23.8H, v19.8H, v3.H[7]
    mla v24.8H, v19.8H, v4.H[7]
    mla v25.8H, v19.8H, v5.H[7]
    mla v26.8H, v19.8H, v6.H[7]
    mla v27.8H, v19.8H, v7.H[7]
    mla v28.8H, v19.8H, v8.H[7]
    mla v29.8H, v19.8H, v9.H[7]
    mla v30.8H, v19.8H, v10.H[7]
    mla v31.8H, v19.8H, v11.H[7]

    add x1, x1, #16
    lsl w12, w15, #4
    add x2, x2, x12

    sub w10, w10, #8
    cbnz w10, _ijk_mla_loopk

// ================

    lsl w12, w15, #1
    add x25, x0, x12
    add x26, x25, x12
    add x27, x26, x12

    str q20, [x0]
    str q21, [x25]
    str q22, [x26]
    str q23, [x27]

    lsl w12, w15, #3
    str q24, [x0, x12]
    str q25, [x25, x12]
    str q26, [x26, x12]
    str q27, [x27, x12]

    lsl w12, w15, #4
    str q28, [x0, x12]
    str q29, [x25, x12]
    str q30, [x26, x12]
    str q31, [x27, x12]

    lsl w12, w11, #1
    sub x1, x1, x12

    mul w12, w15, w11
    lsl w12, w12, #1
    sub x2, x2, x12

    add x0, x0, #16
    add x2, x2, #16

    sub w9, w9, #8
    cbnz w9, _ijk_mla_loopj

// ================

    lsl w12, w15, #1
    sub x2, x2, x12

    lsl w12, w15, #1
    sub x0, x0, x12
    mov w13, #24
    mul w12, w15, w13
    add x0, x0, x12
    add x1, x1, x12

    sub w7, w7, #12
    cbnz w7, _ijk_mla_loopi

// ================

    pop_all

    br lr







