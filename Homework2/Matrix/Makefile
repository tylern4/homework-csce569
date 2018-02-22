UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	FOPENMP = -fopenmp
endif
CC = gcc
CCFLAGS = -w -g $(FOPENMP)
LINKS = -lm
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -O0 $(CCFLAGS) -o $@_0 $< $(LINKS)
	$(CC) -O3 $(CCFLAGS) -o $@_3 $< $(LINKS)

clean:
	rm -f *_0 *_3
