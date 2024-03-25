OUTPUT_FOLDER = bin

all: serial parallel

parallel-mpi:
	@mpicc src/matrix.c src/open-mpi/mpi.c -o $(OUTPUT_FOLDER)/mpi

run-parallel-mpi:
	@mpirun -n 4 ./$(OUTPUT_FOLDER)/mpi

serial:
	@g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial

run-serial:
	@./$(OUTPUT_FOLDER)/serial