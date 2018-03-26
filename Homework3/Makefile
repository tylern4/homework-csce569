CC = mpicc
CCFLAGS = -w
LINKS = -lm
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: $(PROG)

$(PROG): %: %.c
	$(CC) -O3 $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
