CC = gcc
CCFLAGS = -w -g
LINKS = -lm
#-fopenmp
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -O0 $(CCFLAGS) -o $@_0 $< $(LINKS)
	$(CC) -O3 $(CCFLAGS) -o $@_3 $< $(LINKS)

fast:
	/opt/intel/bin/icc -fast -xSSE4.2 -m64 $(CCFLAGS) -o $@_fast $< $(LINKS)

clean:
	rm -f mm_0 mm_3 matvec_0 matvec_3 jacobi_0 jacobi_3
