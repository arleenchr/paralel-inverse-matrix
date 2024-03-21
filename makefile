OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpicc src/matrix.c src/open-mpi/mpi.c -o bin/mpi

run:
	mpirun -n 4 ./bin/mpi

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial