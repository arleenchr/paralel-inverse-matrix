// mpicc src/matrix.c src/open-mpi/mpi.c -o src/open-mpi/mpi.out

#include <mpi.h>
#include <stdio.h>
#include "../matrix.h"

void distributeMatrix(struct Matrix *matrix, struct Matrix *identity);
void gatherMatrix(struct Matrix *matrix, struct Matrix *identity);
void gaussianElimination(struct Matrix *matrix, struct Matrix *identity);

int main(void) {
    struct Matrix inputMatrix;
    struct Matrix identityMatrix;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (world_rank == 0) {
        printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
        
        /* Init matrix */
        char filename[256];
        scanf("%255s", filename);
        readMatrixFromFile(filename, &inputMatrix);
        // printMatrix(&inputMatrix);
        
        identityMatrix.size = inputMatrix.size;
        createIdentityMatrix(&identityMatrix);
        // printMatrix(&identityMatrix);


        // Partial pivoting
        double d;
        for(int i = inputMatrix.size; i > 1; --i)
        {
            if(inputMatrix.buffer[i-1][1] < inputMatrix.buffer[i][1])
            {
                for(int j = 0; j < inputMatrix.size; ++j)
                {
                    d = inputMatrix.buffer[i][j];
                    inputMatrix.buffer[i][j] = inputMatrix.buffer[i-1][j];
                    inputMatrix.buffer[i-1][j] = d;

                    d = identityMatrix.buffer[i][j];
                    identityMatrix.buffer[i][j] = identityMatrix.buffer[i-1][j];
                    identityMatrix.buffer[i-1][j] = d;
                }
            }
        }


    }

    // Distribute matrix to all processes
    distributeMatrix(&inputMatrix, &identityMatrix);

    // Perform Gaussian elimination in parallel
    gaussianElimination(&inputMatrix, &identityMatrix);

    // Gather the results
    gatherMatrix(&inputMatrix, &identityMatrix);

    MPI_Finalize();

    printMatrix(&identityMatrix);

    return 0;
}

// Function to distribute the matrix to all processes
void distributeMatrix(struct Matrix *matrix, struct Matrix *identity) {
    MPI_Bcast(&(matrix->size), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(matrix->buffer[0][0]), matrix->size * matrix->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(identity->buffer[0][0]), identity->size * identity->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Function to gather the matrix from all processes
void gatherMatrix(struct Matrix *matrix, struct Matrix *identity) {
    MPI_Gather(MPI_IN_PLACE, matrix->size * matrix->size, MPI_DOUBLE, &(matrix->buffer[0][0]), matrix->size * matrix->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(MPI_IN_PLACE, identity->size * identity->size, MPI_DOUBLE, &(identity->buffer[0][0]), identity->size * identity->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Function to perform Gaussian elimination in parallel
void gaussianElimination(struct Matrix *matrix, struct Matrix *identity) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    for (int k = 0; k < matrix->size; k++) {
        // Broadcast pivot row to all processes
        MPI_Bcast(&(matrix->buffer[k][0]), matrix->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Parallelize the elimination step
        for (int i = world_rank; i < matrix->size; i += world_size) {
            if (i != k) {
                double factor = matrix->buffer[i][k] / matrix->buffer[k][k];
                for (int j = k; j < matrix->size; j++) {
                    matrix->buffer[i][j] -= factor * matrix->buffer[k][j];
                    identity->buffer[i][j] -= factor * identity->buffer[k][j];
                }
            }
        }
    }
}
