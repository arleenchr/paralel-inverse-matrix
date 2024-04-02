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


parallel-cobain:
	@gcc $(SOURCE_FOLDER)/matrix.c $(MP_FOLDER)/cobain.c --openmp -o $(OUTPUT_FOLDER)/cobain

run-parallel-cobain:
	@./$(OUTPUT_FOLDER)/cobain

serial-cobain:
	@gcc $(SOURCE_FOLDER)/matrix.c $(MP_FOLDER)/serial.c -o $(OUTPUT_FOLDER)/serialcobain

run-serial-cobain:
	@./$(OUTPUT_FOLDER)/serialcobain

	

serial:
	@g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial

run-serial:
	@./$(OUTPUT_FOLDER)/serial