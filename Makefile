CC=gcc
CFLAGS=-O0 -g

LNK=$(CC)
LNKFLAGS=-lm

test-nn-obj-deps=libnn.o test-nn.o

all: test-nn

libnn.o : libnn.c libnn.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

test-nn.o : test-nn.c libnn.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

test-nn : $(test-nn-obj-deps)
	$(LNK) $(LNKFLAGS) $(test-nn-obj-deps)  -o $@

clean :
	@rm -f test-nn $(test-nn-obj-deps)
