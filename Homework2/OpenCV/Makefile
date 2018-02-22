CC = g++
ifeq ($(shell hostname), workstation)
OPENCV_FLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LINK = $(shell pkg-config --libs opencv) -fopenmp
else
OPENCV_FLAGS = $(shell pkg-config --cflags opencv-3.2.0-dev)
OPENCV_LINK = $(shell pkg-config --libs opencv-3.2.0-dev) -fopenmp
endif

CCFLAGS = -O3 -w -g $(OPENCV_FLAGS)
LINKS = -lm $(OPENCV_LINK)
PROG = $(patsubst %.cpp,%,$(wildcard *.cpp))
PROG = opencv_smooth opencv_hist

.PHONY: all clean

all: clean $(PROG)

$(PROG): %: %.cpp
	$(CC) $(CCFLAGS) -o $@ $< $(LINKS)

clean:
	rm -f $(PROG)
