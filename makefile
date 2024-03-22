OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	@mpicc src/matrix.c src/open-mpi/mpi.c -o $(OUTPUT_FOLDER)/mpi

run-parallel:
	@mpirun -n 4 ./$(OUTPUT_FOLDER)/mpi

serial:
	@g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial

run-serial:
	@./$(OUTPUT_FOLDER)/serial