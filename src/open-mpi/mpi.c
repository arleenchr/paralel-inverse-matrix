// mpicc src/matrix.c src/open-mpi/mpi.c -o src/open-mpi/mpi.out

#include <mpi.h>
#include <stdio.h>
#include "../matrix.h"

void distributeMatrix(struct Matrix *matrix, struct Matrix *identity);
void gatherMatrix(struct Matrix *matrix, struct Matrix *identity);
void gaussianElimination(struct Matrix *matrix, struct Matrix *identity);

int main(void) {
    Matrix inputMatrix;
    Matrix identityMatrix;
    Matrix procInputMatrix;
    Matrix procIdentityMatrix;
    int inputMatrixCol;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (world_rank == 0) {
        /* Init matrix */
        char filename[256];
        scanf("%255s", filename);

        inputMatrix = readMatrixFromFile(filename);
        printMatrix(inputMatrix);
        printf("-------\n");

        identityMatrix = createIdentityMatrix(inputMatrix.col);
        printMatrix(identityMatrix);

        MPI_Bcast(&inputMatrix.col, 1, MPI_INT, 0, MPI_COMM_WORLD);
        inputMatrixCol = inputMatrix.col;
        // MPI_Recv(&matrix_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // for (int i=1; i<world_size; i++){
        //     MPI_Send(&inputMatrix.size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        // }

    } else {
        MPI_Bcast(&inputMatrixCol, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // MPI_Recv(&matrix_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   
    procInputMatrix = createMatrix(inputMatrixCol / world_size, inputMatrixCol);
    procIdentityMatrix = createMatrix(inputMatrixCol / world_size, inputMatrixCol);
    // printf("inputMatrixCol process %d = %d\n", world_rank, inputMatrixCol);
    printf("input matrix size process %d = (%d x %d)\n", world_rank, procInputMatrix.row, procInputMatrix.col);

    MPI_Scatter(
        inputMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        procInputMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        0, MPI_COMM_WORLD);
    MPI_Scatter(
        identityMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        procIdentityMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        0, MPI_COMM_WORLD);

    printf("procInputMatrix from proc%d\n", world_rank);
    printMatrix(procInputMatrix);
    printMatrix(procIdentityMatrix);

    MPI_Finalize();

    freeMatrix(&procInputMatrix);

    if (world_rank == 0) {
        freeMatrix(&inputMatrix);
        freeMatrix(&identityMatrix);
    }

    return 0;
}

// // Function to distribute the matrix to all processes
// void distributeMatrix(struct Matrix *matrix, struct Matrix *identity) {
//     MPI_Bcast(&(matrix->size), 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&(matrix->buffer[0][0]), matrix->size * matrix->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     // MPI_Bcast(&(identity->buffer[0][0]), identity->size * identity->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// }

// // Function to gather the matrix from all processes
// void gatherMatrix(struct Matrix *matrix, struct Matrix *identity) {
//     MPI_Gather(MPI_IN_PLACE, matrix->size * matrix->size, MPI_DOUBLE, &(matrix->buffer[0][0]), matrix->size * matrix->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     // MPI_Gather(MPI_IN_PLACE, identity->size * identity->size, MPI_DOUBLE, &(identity->buffer[0][0]), identity->size * identity->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// }

// // Function to perform Gaussian elimination in parallel
// void gaussianElimination(struct Matrix *matrix, struct Matrix *identity) {
//     int world_rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
//     int world_size;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
//     for (int k = 0; k < matrix->size; k++) {
//         // Broadcast pivot row to all processes
//         MPI_Bcast(&(matrix->buffer[k][0]), matrix->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
//         // Parallelize the elimination step
//         for (int i = world_rank; i < matrix->size; i += world_size) {
//             if (i != k) {
//                 double factor = matrix->buffer[i][k] / matrix->buffer[k][k];
//                 for (int j = k; j < matrix->size; j++) {
//                     matrix->buffer[i][j] -= factor * matrix->buffer[k][j];
//                     // identity->buffer[i][j] -= factor * identity->buffer[k][j];
//                 }
//             }
//         }
//     }
// }
