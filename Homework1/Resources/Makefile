CC = gcc
CCFLAGS = -O0 -w -g 
LINKS = -lm -fopenmp
PROG = $(patsubst %.c,%,$(wildcard *.c))

.PHONY: all clean

all: clean $(PROG)

$(PROG): %: %.c
	$(CC) $(CCFLAGS) -o $@ $< $(LINKS)

clean: 
	rm -f $(PROG)
