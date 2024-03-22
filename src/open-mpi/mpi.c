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

double* getRow (Matrix m, size_t rowNum){
    double* row = (double *)calloc(m.col, sizeof(double));
    memcpy(row, &(m.buffer[rowNum * m.row]), m.row * sizeof(double));
    return row;
}

void swapRow(Matrix* matrix, int row1, int row2){
    if(row1 < 0 || row1 >= matrix->row || row2 < 0 || row2 >= matrix->row){
        fprintf(stderr, "Invalid row indices\n");
        return;
    }

    int startIndexRow2 = row2 * matrix->col;
    int startIndexRow1 = row1 * matrix->col;

    double* temp_row = (double*)malloc(matrix->col * sizeof(double));
    if(temp_row == NULL){
        fprintf(stderr, "Memory allocation failed\n");
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
        // char filename[256];
        // scanf("%255s", filename);       

        inputMatrix = readMatrixFromFile();
        // printMatrix(inputMatrix);
        // printf("-------\n");

        identityMatrix = createIdentityMatrix(inputMatrix.col);
        // printMatrix(identityMatrix);
        // printf("-------\n");
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
                for (size_t swapIdx = i+1; swapIdx < size; swapIdx++){
                    if (colBuffer[swapIdx] != 0.){
                        swapRow(&inputMatrix, i, swapIdx);
                        swapRow(&identityMatrix, i, swapIdx);
                        // printf("swap row %zu & %zu\n", i, swapIdx);
                        break;
                    } else if (swapIdx == size - 1) {
                        invertible = false;
                        fprintf(stderr, "Matrix can not be inversed.\n");
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
        
        // free inputRowToBeDivided & outputRowToBeDivided
        free(inputRowToBeDivided);
        free(outputRowToBeDivided);

        // if (world_rank == 0){
        //     printf("Input Matrix:\n");
        //     printMatrix(inputMatrix);
        //     printf("Inversed Matrix:\n");
        //     printMatrix(identityMatrix);
        // }

        /* Eliminating */
        // printf("\n===== ELIMINATING =====\n");
        double* inputPivotRow = (double *)calloc(size, sizeof(double));
        double* outputPivotRow = (double *)calloc(size, sizeof(double));

        if (world_rank == 0) {
            inputPivotRow = getRow(inputMatrix, i);
            outputPivotRow = getRow(identityMatrix, i);
        }
        MPI_Bcast(inputPivotRow, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(outputPivotRow, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // printf("input pivot row i = %zu, proc%d\n", i, world_rank);
        // printRow(inputPivotRow,size);
        // printf("output pivot row i = %zu, proc%d\n", i, world_rank);
        // printRow(outputPivotRow,size);

        /* Scatter the inputMatrix to all processes for elimination */
        MPI_Scatter(
            inputMatrix.buffer, nRow * size, MPI_DOUBLE, 
            procInputMatrix.buffer, nRow * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(
            identityMatrix.buffer, nRow * size, MPI_DOUBLE, 
            procOutputMatrix.buffer, nRow * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // printf("input matrix proc%d\n", world_rank);
        // printMatrix(procInputMatrix);
        // printf("output matrix proc%d\n", world_rank);
        // printMatrix(procOutputMatrix);
        // MPI_Barrier(MPI_COMM_WORLD);

        for (size_t localRow = 0; localRow < nRow; localRow++){
            if (localRow != i - world_rank * nRow){
                // eliminate each row
                double eliminateFactor = procInputMatrix.buffer[localRow * size + i];

                // subtract
                for (size_t col = 0; col < size; col++){
                    // printf("SUBTRACT col=%zu\n", col);
                    procInputMatrix.buffer[localRow * size + col] -=
                        inputPivotRow[col] * eliminateFactor;
                    procOutputMatrix.buffer[localRow * size + col] -=
                        outputPivotRow[col] * eliminateFactor;
                }
                
                // printf("\nPROCESS %d eliminating\nInput Matrix:\n", world_rank);
                // printMatrix(procInputMatrix);
                // printf("Output Matrix:\n");
                // printMatrix(procOutputMatrix);
            }
        }

        MPI_Gather(
            procInputMatrix.buffer, nRow * size, MPI_DOUBLE,
            inputMatrix.buffer, nRow * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(
            procOutputMatrix.buffer, nRow * size, MPI_DOUBLE,
            identityMatrix.buffer, nRow * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        free(inputPivotRow);
        free(outputPivotRow);

    }

    freeMatrix(&procInputMatrix);
    freeMatrix(&procOutputMatrix);

    if (world_rank == 0){
        // printf("Input Matrix:\n");
        // printMatrix(inputMatrix);
        printf("%d\n",size);
        printMatrix(identityMatrix);
        freeMatrix(&inputMatrix);
        freeMatrix(&identityMatrix);
    }
    
    MPI_Finalize();

    return 0;
}