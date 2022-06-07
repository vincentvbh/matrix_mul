# Matrix Multiplications

## Basic Information
- Apple M1
- `g++-11` with `-O3`
- Strassen's with 7 multiplications and 18 additions
- Winograd's improvement with 7 multiplications and 15 additions. There is an error in the wiki, we compute the following:
    - $$\begin{pmatrix} A_{0, 0} & A_{0, 1} \\ A_{1, 0} & A_{1, 1}  \end{pmatrix}$$


# 48 x 2
## Strassen's
```
ikj SIMD asm Dense cycles:
477559
preprocessing cycles:
25575
postprocessing cycles:
13903
Strassen cycles:
495479
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
477421
preprocessing cycles:
25705
postprocessing cycles:
13368
Strassen_Winograd cycles:
476667
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
477378
A preprocessing cycles:
12873
postprocessing cycles:
13150
Strassen_Winograd with preprocessing cycles:
485234
```
##

# 48 x 3
## Strassen's
```
ikj SIMD asm Dense cycles:
1599693
preprocessing cycles:
97516
postprocessing cycles:
29584
Strassen cycles:
1623784
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
1599005
preprocessing cycles:
97267
postprocessing cycles:
29720
Strassen_Winograd cycles:
1554038
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
1599877
A preprocessing cycles:
29399
postprocessing cycles:
29620
Strassen_Winograd with preprocessing cycles:
1599976
```
##

# 48 x 4
## Strassen's
```
ikj SIMD asm Dense cycles:
3781084
preprocessing cycles:
172022
postprocessing cycles:
53134
Strassen cycles:
3791761
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
3780530
preprocessing cycles:
172270
postprocessing cycles:
53122
Strassen_Winograd cycles:
3637410
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
3779235
A preprocessing cycles:
72761
postprocessing cycles:
52758
Strassen_Winograd with preprocessing cycles:
3707514
```
##

# 48 x 5
## Strassen's
```
ikj SIMD asm Dense cycles:
7368329
preprocessing cycles:
267788
postprocessing cycles:
82184
Strassen cycles:
7261482
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
7368151
preprocessing cycles:
268470
postprocessing cycles:
82786
Strassen_Winograd cycles:
6992363
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
7369256
A preprocessing cycles:
134824
postprocessing cycles:
82246
Strassen_Winograd with preprocessing cycles:
7113471
```
##

# 48 x 6
## Strassen's
```
ikj SIMD asm Dense cycles:
12739991
preprocessing cycles:
396038
postprocessing cycles:
118667
Strassen cycles:
12357018
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
12741915
preprocessing cycles:
387324
postprocessing cycles:
118608
Strassen_Winograd cycles:
11953831
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
12748408
A preprocessing cycles:
198431
postprocessing cycles:
118951
Strassen_Winograd with preprocessing cycles:
12115112
```
##

# 48 x 7
## Strassen's
```
ikj SIMD asm Dense cycles:
20183786
preprocessing cycles:
533577
postprocessing cycles:
162126
Strassen cycles:
19382420
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
20159568
preprocessing cycles:
534372
postprocessing cycles:
161470
Strassen_Winograd cycles:
18769284
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
20165321
A preprocessing cycles:
266913
postprocessing cycles:
160093
Strassen_Winograd with preprocessing cycles:
18959373
```
##

# 48 x 8
## Strassen's
```
ikj SIMD asm Dense cycles:
30388557
preprocessing cycles:
723981
postprocessing cycles:
198577
Strassen cycles:
28599952
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
30356495
preprocessing cycles:
702586
postprocessing cycles:
198006
Strassen_Winograd cycles:
27829933
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
30350264
A preprocessing cycles:
349953
postprocessing cycles:
198373
Strassen_Winograd with preprocessing cycles:
28102656
```
##

# 48 x 15
## Strassen's
```
ikj SIMD asm Dense cycles:
200790896
preprocessing cycles:
2473064
postprocessing cycles:
736945
Strassen cycles:
181316418
```

## Winograd's form (lower additions/subtractions)
```
ikj SIMD asm Dense cycles:
200726021
preprocessing cycles:
2474804
postprocessing cycles:
738436
Strassen_Winograd cycles:
178545069
```

## Winograd's form with preprocessing B
```
ikj SIMD asm Dense cycles:
201102271
A preprocessing cycles:
1229494
postprocessing cycles:
737946
Strassen_Winograd with preprocessing cycles:
179537165
```
##

# Comparing to Eigen

## (96, 144, 120)
```
ikj SIMD asm Dense cycles:
901087
ikj SIMD intrinsics Dense cycles:
1262581
ijk SIMD asm Dense cycles:
895292
ijk SIMD intrinsics Dense cycles:
906066
Eigen Dense cycles:
1283077
non-zeros: 4948, 5001
Eigen Sparse cycles:
27708745
Eigen Condensed Sparse cycles:
25453986
Eigen Sparse Sparse cycles:
13062360
```

## (144, 120, 168)
```
ikj SIMD asm Dense cycles:
1567361
ikj SIMD intrinsics Dense cycles:
2133731
ijk SIMD asm Dense cycles:
1570237
ijk SIMD intrinsics Dense cycles:
1583747
Eigen Dense cycles:
2277661
non-zeros: 5015, 4993
Eigen Sparse cycles:
28133718
Eigen Condensed Sparse cycles:
25852076
Eigen Sparse Sparse cycles:
12813585
```



