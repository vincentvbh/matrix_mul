
CC = g++-11

HEADERs = m1cycles.h params.h tools.h

SOURCEs = asm_f.S m1cycles.c tools.c

CFLAGS = -O3 -mcpu=native -mtune=native -ftree-vectorize

EIGEN_PATH = /Users/vincent/Desktop/git/public/eigen

all: strassen strassen_winograd strassen_winograd_pre eigen_test

strassen: $(HEADERs) $(SOURCEs) strassen.c
	$(CC) -o strassen strassen.c $(SOURCEs) $(CFLAGS)

strassen_winograd: $(HEADERs) $(SOURCEs) strassen_winograd.c
	$(CC) -o strassen_winograd strassen_winograd.c $(SOURCEs) $(CFLAGS)

strassen_winograd_pre: $(HEADERs) $(SOURCEs) strassen_winograd_pre.c
	$(CC) -o strassen_winograd_pre strassen_winograd_pre.c $(SOURCEs) $(CFLAGS)

eigen_test: $(HEADERs) $(SOURCEs) eigen_test.c
	$(CC) -o eigen_test eigen_test.c $(SOURCEs) $(CFLAGS) -I $(EIGEN_PATH)

clean:
	rm -f strassen
	rm -f strassen_winograd
	rm -f strassen_winograd_pre
	rm -f eigen_test


