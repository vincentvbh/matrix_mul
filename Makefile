
CC = g++-11

HEADERs = m1cycles.h params.h tools.h

SOURCEs = asm_f.S m1cycles.c tools.c

CFLAGS = -O3 -mcpu=native -mtune=native -ftree-vectorize

EIGEN_PATH = /Users/vincent/Desktop/git/public/eigen

all: strassen winograd winograd_pre eigen_test

strassen: $(HEADERs) $(SOURCEs) strassen.c
	$(CC) -o strassen strassen.c $(SOURCEs) $(CFLAGS)

winograd: $(HEADERs) $(SOURCEs) winograd.c
	$(CC) -o winograd winograd.c $(SOURCEs) $(CFLAGS)

winograd_pre: $(HEADERs) $(SOURCEs) winograd_pre.c
	$(CC) -o winograd_pre winograd_pre.c $(SOURCEs) $(CFLAGS)

eigen_test: $(HEADERs) $(SOURCEs) eigen_test.c
	$(CC) -o eigen_test eigen_test.c $(SOURCEs) $(CFLAGS) -I $(EIGEN_PATH)

clean:
	rm -f strassen
	rm -f winograd
	rm -f winograd_pre
	rm -f eigen_test


