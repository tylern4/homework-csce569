CC = gcc
CCFLAGS = -w
LINKS = -lm -fopenmp
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -O0 $(CCFLAGS) -o $@_0 $< $(LINKS)
	$(CC) -O3 $(CCFLAGS) -o $@_3 $< $(LINKS)

clean:
	rm -rf *_0 *_3 *.dSYM
