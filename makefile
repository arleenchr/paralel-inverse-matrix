OUTPUT_FOLDER = bin
SOURCE_FOLDER = src
MPI_FOLDER = src/open-mpi
MP_FOLDER = src/open-mp

all: serial parallel

parallel-mpi:
	@mpicc $(SOURCE_FOLDER)/matrix.c $(MPI_FOLDER)/mpi.c -o $(OUTPUT_FOLDER)/mpi

run-parallel-mpi:
	@mpirun -n 4 ./$(OUTPUT_FOLDER)/mpi

parallel-mp:
	@gcc $(SOURCE_FOLDER)/matrix.c $(MP_FOLDER)/mp.c --openmp -o $(OUTPUT_FOLDER)/mp

run-parallel-mp:
	@./$(OUTPUT_FOLDER)/mp


serial-c:
	@gcc $(SOURCE_FOLDER)/matrix.c $(MP_FOLDER)/serial.c -o $(OUTPUT_FOLDER)/serialc

run-serial-c:
	@./$(OUTPUT_FOLDER)/serialc

serial:
	@g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial

run-serial:
	@./$(OUTPUT_FOLDER)/serial