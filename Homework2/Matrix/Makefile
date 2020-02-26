UNAME := $(shell uname)
FOPENMP = -fopenmp
#CC = gcc
CCFLAGS = -w -g $(FOPENMP)
LINKS = -lm
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -Ofast $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
