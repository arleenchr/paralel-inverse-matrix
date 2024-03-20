OUTPUT_FOLDER = bin

all: serial parallel

parallel:
# TODO : Parallel compilation
	mpicc src/matrix.c src/open-mpi/mpi.c -o src/open-mpi/mpi

run:
	mpirun -n 4 ./src/open-mpi/mpi

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial