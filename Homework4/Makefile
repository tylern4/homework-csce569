CC = nvcc
CCFLAGS = -arch=sm_52 -gencode=arch=compute_52,code=sm_52 -Xcompiler -fopenmp
LINKS = -lm -lpthread -lcublas
PROG = matmul
#$(patsubst %.cu,%,$(wildcard *.cu))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.cu
	$(CC) -O3 $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
