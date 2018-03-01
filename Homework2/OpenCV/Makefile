CC = g++
ifeq ($(shell hostname), cocsce-l3d22-04)
OPENCV_FLAGS = $(shell pkg-config --cflags opencv-3.2.0-dev)
OPENCV_LINK = $(shell pkg-config --libs opencv-3.2.0-dev) -fopenmp
else
OPENCV_FLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LINK = $(shell pkg-config --libs opencv) -fopenmp
endif

CCFLAGS = -O3 -w -g -std=c++17 $(OPENCV_FLAGS)
LINKS = -lm $(OPENCV_LINK)
PROG = $(patsubst %.cpp,%,$(wildcard *.cpp))

.PHONY: all clean

all: clean $(PROG)

$(PROG): %: %.cpp
	$(CC) $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
