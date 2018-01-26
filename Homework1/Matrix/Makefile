CC = gcc
CCFLAGS = -w -g
LINKS = -lm -fopenmp
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -O0 $(CCFLAGS) -o $@_0 $< $(LINKS)
	$(CC) -O3 $(CCFLAGS) -o $@_3 $< $(LINKS)

clean:
	rm -f mm_0 mm_3 matvec_0 matvec_3
