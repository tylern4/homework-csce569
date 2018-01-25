CC = g++
CCFLAGS = -O0 -w -g $(shell pkg-config --cflags opencv-3.2.0-dev)
LINKS = -lm $(shell pkg-config --libs opencv-3.2.0-dev)
PROG = $(patsubst %.cpp,%,$(wildcard *.cpp))

.PHONY: all clean

all: clean $(PROG)

$(PROG): %: %.cpp
	$(CC) $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
