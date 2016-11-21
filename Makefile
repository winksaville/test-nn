# Makefile for test-nn
# Parameters:
#   DBG=0 or 1 (default = 0)

# _DBG will be 0 if DBG isn't defined on the command line
_DBG = +$(DBG)
ifeq ($(_DBG), +)
  _DBG = 0
endif

# Make sure _EPOCH_COUNT is at least 1
_EPOCH_COUNT = +$(EPOCH_COUNT)
ifeq ($(_EPOCH_COUNT), +)
  _EPOCH_COUNT = 1
endif

CC=gcc
CFLAGS=-O0 -g -DDBG=$(_DBG) -DEPOCH_COUNT=$(_EPOCH_COUNT)

LNK=$(CC)
LNKFLAGS=-lm

test-nn-obj-deps=NeuralNet.o NeuralNetIo.o test-nn.o

all: test-nn

NeuralNetIo.o : NeuralNetIo.c NeuralNet.h NeuralNetIo.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

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
