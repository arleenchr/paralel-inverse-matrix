// mpicc mpi.c -o mpi

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "../matrix.h"

void printRow (double* row, int column){
    for (int i=0; i<column; i++){
        printf("%.2f ", row[i]);
    }
    printf("\n");
}

void swapRow(Matrix* matrix, int row1, int row2){
    if(row1 < 0 || row1 >= matrix->row || row2 < 0 || row2 >= matrix->row){
        printf("Invalid row indices\n");
        return;
    }

    int startIndexRow2 = row2 * matrix->col;
    int startIndexRow1 = row1 * matrix->col;

    double* temp_row = (double*)malloc(matrix->col * sizeof(double));
    if(temp_row == NULL){
        printf("Memory allocation failed\n");
        return;
    }

    // Copy row1 to temp_row
    memcpy(temp_row, &(matrix->buffer[startIndexRow1]), matrix->col * sizeof(double));

    // Copy row2 to row1
    memcpy(&(matrix->buffer[startIndexRow1]), &(matrix->buffer[startIndexRow2]), matrix->col * sizeof(double));

    // Copy temp_row to row2
    memcpy(&(matrix->buffer[startIndexRow2]), temp_row, matrix->col * sizeof(double));

    free(temp_row);
}

int main(void) {
    Matrix inputMatrix;
    Matrix identityMatrix;
    int size;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (world_rank == 0){
        /* Init matrix */
        char filename[256];
        scanf("%255s", filename);

        inputMatrix = readMatrixFromFile(filename);
        printMatrix(inputMatrix);
        printf("-------\n");

        identityMatrix = createIdentityMatrix(inputMatrix.col);
        printMatrix(identityMatrix);
        printf("-------\n");
        size = inputMatrix.col;
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Define variables */
    size_t nRow = size / world_size;
    bool invertible = true; // to check if the matrix can be inverted or not
    Matrix procInputMatrix = createMatrix(nRow, size);
    Matrix procOutputMatrix = createMatrix(nRow, size);

    // iterate every row
    for (size_t i = 0; i < size; i++){
        /* Partial Pivoting */
        /* Swap-swapan */
        if (world_rank == 0){
            double* colBuffer = getColFromMatrix(inputMatrix, i);
            if (colBuffer[i] == 0.){
                // swap rows
                // search for the nearest non-zero row
                for (size_t swapIdx = i; swapIdx < size; swapIdx++){
                    if (colBuffer[swapIdx] != 0.){
                        swapRow(&inputMatrix, i, swapIdx);
                        swapRow(&identityMatrix, i, swapIdx);
                        // printf("swap row %zu & %zu\n", i, swapIdx);
                        break;
                    } else {
                        invertible = false;
                        printf("Matrix can not be inversed.\n");
                    }
                }          
            }
        }
        MPI_Bcast(&invertible, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        if (!invertible) {
            exit(1);
        }

        // The matrix is now can be inverted
        /* Divide the row by the pivot factor */
        double* inputRowToBeDivided = (double *)calloc(size, sizeof(double));
        double* outputRowToBeDivided = (double *)calloc(size, sizeof(double));
        double pivotFactor;
        if (world_rank == 0){
            pivotFactor = inputMatrix.buffer[i * size + i];
        }
        MPI_Bcast(&pivotFactor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatter(
            inputMatrix.buffer + i * size, nRow, MPI_DOUBLE, 
            inputRowToBeDivided, nRow, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(
            identityMatrix.buffer + i * size, nRow, MPI_DOUBLE, 
            outputRowToBeDivided, nRow, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Divide
        for (int j = 0; j < nRow; j++){
            inputRowToBeDivided[j] /= pivotFactor;
            outputRowToBeDivided[j] /= pivotFactor;
        }

        MPI_Gather(
            inputRowToBeDivided, nRow, MPI_DOUBLE,
            inputMatrix.buffer + i * size, nRow, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(
            outputRowToBeDivided, nRow, MPI_DOUBLE,
            identityMatrix.buffer + i * size, nRow, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (world_rank == 0){
            printf("Input Matrix:\n");
            printMatrix(inputMatrix);
            printf("Inversed Matrix:\n");
            printMatrix(identityMatrix);
        }
    }
    
    
    MPI_Finalize();

    return 0;
}