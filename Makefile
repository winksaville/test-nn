# Makefile for test-nn
# Parameters:
#   DBG=0 or 1 (default = 0)

# _DBG will be 0 if DBG isn't defined on the command line
_DBG = +$(DBG)
ifeq ($(_DBG), +)
  _DBG = 0
endif

P1=10000000
OUTPUT=out.txt

CC=clang
CFLAGS=-O3 -g -Weverything -Werror -DDBG=$(_DBG)
OD=objdump
ODFLAGS=-S -M x86_64,intel

LNK=$(CC)
LNKFLAGS=-lm

test-nn-obj-deps=NeuralNet.o NeuralNetIo.o rand0_1.o test-nn.o

all: test-nn

rand0_1.o : rand0_1.c NeuralNet.h NeuralNetIo.h rand0_1.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

NeuralNetIo.o : NeuralNetIo.c NeuralNet.h NeuralNetIo.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

NeuralNet.o : NeuralNet.c NeuralNet.h rand0_1.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

test-nn.o : test-nn.c NeuralNet.h rand0_1.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

test-nn : $(test-nn-obj-deps)
	$(LNK) $(LNKFLAGS) $(test-nn-obj-deps)  -o $@
	$(OD) $(ODFLAGS) $@ > $@.asm

test: test-nn
	./test-nn $(P1)

clean :
	@rm -f test-nn $(test-nn-obj-deps) *.asm
