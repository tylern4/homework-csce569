CC = nvcc
CCFLAGS = -w -Xcompiler -fopenmp
LINKS = -lm -lpthread -lcublas
PROG = $(patsubst %.cu,%,$(wildcard *.cu))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.cu
	$(CC) -O3 $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
