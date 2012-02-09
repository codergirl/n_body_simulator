#makefile
CFLAGS = -g -O4 -std=c99

all: 
	mpicc -std=c99 ring.c -lmpe -lm  -o ring -lX11

clean:
	rm -rf *.o 


