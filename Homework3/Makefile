UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	FOPENMP = -fopenmp
endif
CC = mpicc
CCFLAGS = -w -g $(FOPENMP)
LINKS = -lm
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -O3 $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG) 
