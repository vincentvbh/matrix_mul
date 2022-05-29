
CC = g++-11

HEADERs = m1cycles.h params.h

SOURCEs = asm_f.S m1cycles.c

CFLAGS = -O3 -mcpu=native -mtune=native -ftree-vectorize -I /Users/vincent/Desktop/git/public/eigen

all: test eigen_test

test: $(HEADERs) $(SOURCEs) test.c
	$(CC) -o test test.c $(SOURCEs) $(CFLAGS)

eigen_test: $(HEADERs) $(SOURCEs) eigen_test.c
	$(CC) -o eigen_test eigen_test.c $(SOURCEs) $(CFLAGS)

clean:
	rm -f test


