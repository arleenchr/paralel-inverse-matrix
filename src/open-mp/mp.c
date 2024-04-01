// gcc mp.c --openmp -o mp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "../matrix.h"

/*Getter: row in a certain index*/
double* getRow (Matrix m, size_t rowNum){
    double* row = (double *)calloc(m.col, sizeof(double));
    memcpy(row, &(m.buffer[rowNum * m.row]), m.row * sizeof(double));
    return row;
}

/*Swap row with another row*/
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
    // #pragma omp parallel num_threads(8) 
    // {
    //     int nthreads, tid;
    //     nthreads = omp_get_num_threads();
    //     tid = omp_get_thread_num();
    //     printf("Hello world from from thread %d out of %d threads\n", tid, nthreads);
    // }

    /* Init Matrix */
    Matrix inputMatrix;
    Matrix identityMatrix;
    int size;

    inputMatrix = readMatrixFromFile();
    identityMatrix = createIdentityMatrix(inputMatrix.col);
    size = inputMatrix.col;

    bool invertible = true;

    #pragma omp parallel for num_threads(8) shared(inputMatrix, identityMatrix, invertible, size)
    for (size_t i = 0; i < size; i++){
        /* Partial Pivoting */
        /* Swapping indivisible row */
        double* colBuffer = getColFromMatrix(inputMatrix, i);
        if (colBuffer[i] == 0.){
            // Swap rows
            // search for the nearest non-zero row
            for (size_t swapIdx = i+1; swapIdx < size; swapIdx++){
                if (colBuffer[swapIdx] != 0.){
                    #pragma omp critical
                    {
                        swapRow(&inputMatrix, i, swapIdx);
                        swapRow(&identityMatrix, i, swapIdx);
                    }
                    break;
                } else if (swapIdx == size - 1){
                    #pragma omp critical
                    {
                        invertible = false;
                        fprintf(stderr, "Matrix can not be inversed.\n");
                    }
                }
            }      
        }

        // Ensure all threads have checked invertibility before proceeding
        // #pragma omp barrier
        if (!invertible) {
            exit(1);
        }

        // The matrix is now can be inverted
        /* Divide the row by the pivot factor */
        double pivotFactor = inputMatrix.buffer[i * size + i];

        // #pragma omp for
        for (int j = 0; j < size; j++){
            inputMatrix.buffer[i * size + j] /= pivotFactor;
            identityMatrix.buffer[i * size + j] /= pivotFactor;
        }
        
        /* Eliminating */
        // #pragma omp for
        for (size_t j = i + 1; j < size; j++){
            double eliminateFactor = inputMatrix.buffer[j * size + i];
            for (size_t col = 0; col < size; col++){
                inputMatrix.buffer[j * size + col] -= inputMatrix.buffer[i * size + col] * eliminateFactor;
                identityMatrix.buffer[j * size + col] -= identityMatrix.buffer[i * size + col] * eliminateFactor;
            }
        }
    }

    printf("%d\n",size);
    printMatrix(identityMatrix);

    return 0;
}
