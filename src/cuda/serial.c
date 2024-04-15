

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "matrix.h"

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
    Matrix inputMatrix;
    Matrix identityMatrix;
    int size;

    inputMatrix = readMatrixFromFile();
    identityMatrix = createIdentityMatrix(inputMatrix.col);
    size = inputMatrix.col;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / 256;

    bool invertible = true;

    clock_t start = clock();

    for (size_t i = 0; i < size; i++){
        /* Partial Pivoting */
        /* Swapping indivisible row */
        double* colBuffer = getColFromMatrix(inputMatrix, i);

        if (colBuffer[i] == 0.){
            // Swap rows
            // search for the nearest non-zero row
            for (size_t swapIdx = i+1; swapIdx < size; swapIdx++){
                if (colBuffer[swapIdx] != 0.){
                    {
                        swapRow(&inputMatrix, i, swapIdx);
                        swapRow(&identityMatrix, i, swapIdx);
                    }
                    break;
                } else if (swapIdx == size - 1){
                    {
                        invertible = false;
                        fprintf(stderr, "Matrix can not be inversed.\n");
                    }
                }
            }
        }
        // Ensure all threads have checked invertibility before proceeding
        if (!invertible) {
            exit(1);
        }
    }

    printf("Input matrix after pivoting:\n");
    printMatrix(inputMatrix);
    printf("Identity matrix after pivoting:\n");
    printMatrix(identityMatrix);

    /* Eliminating */
    /* Reducing to upper triangle matrix */
    for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
            if (i != j){
                double eliminateFactor = inputMatrix.buffer[j * size + i] / inputMatrix.buffer[i * size + i];
                for (size_t k = 0; k < size; k++){
                    inputMatrix.buffer[j * size + k] -= inputMatrix.buffer[i * size + k] * eliminateFactor;
                    identityMatrix.buffer[j * size + k] -= identityMatrix.buffer[i * size + k] * eliminateFactor;
                }
            }
        }
    }

    printf("Input matrix after eliminate:\n");
    printMatrix(inputMatrix);
    printf("Identity matrix after eliminate:\n");
    printMatrix(identityMatrix);

    /* Reducing to diagonal matrix */
    for (size_t i = 0; i < size; i++){
        /* Divide the row by the pivot factor */
        double pivotFactor = inputMatrix.buffer[i * size + i];

        for (int j = 0; j < size; j++){
            inputMatrix.buffer[i * size + j] /= pivotFactor;
            identityMatrix.buffer[i * size + j] /= pivotFactor;
        }
    }

    printf("Result:\n");
    printf("%d\n",size);
    printMatrix(identityMatrix);

    clock_t end = clock();
    double exectime = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Time taken is %.6f\n", exectime);

    return 0;
}
