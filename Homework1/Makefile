CC = gcc
CCFLAGS = -O0 -march=native -fPIC -fopenmp -w -g 
TARGET = mm
MAINS = $(TARGET).c
MAIN = $(MAINS:%.c=%.o)

.PHONY: all clean

all:	clean $(TARGET)

$(MAIN): %.o : %.c
			$(CC) $(CCFLAGS) -c $< -o $@

$(TARGET): $(MAIN)
		$(CC) $(MAIN) $(CCFLAGS) -o $(TARGET)


clean:
		-rm -f $(TARGET) $(MAIN)
