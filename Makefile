
CC = g++-11

HEADERs = m1cycles.h params.h

SOURCEs = asm_f.S m1cycles.c

CFLAGS = -O3 -mcpu=native -mtune=native -ftree-vectorize -I /Users/vincent/Desktop/git/public/eigen

all: test

test: $(HEADERs) $(SOURCEs) test.c
	$(CC) -o test test.c $(SOURCEs) $(CFLAGS)

clean:
	rm -f test


