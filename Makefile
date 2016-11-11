CC=gcc
CFLAGS=-O0 -g

LNK=$(CC)
LNKFLAGS=-lm

test-nn-obj-deps=NeuralNet.o test-nn.o

all: test-nn

NeuralNet.o : NeuralNet.c NeuralNet.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

test-nn.o : test-nn.c NeuralNet.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

test-nn : $(test-nn-obj-deps)
	$(LNK) $(LNKFLAGS) $(test-nn-obj-deps)  -o $@

test: test-nn
	./test-nn

clean :
	@rm -f test-nn $(test-nn-obj-deps)
