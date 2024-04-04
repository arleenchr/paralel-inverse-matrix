#include <omp.h>
#include <time.h>
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
    Matrix inputMatrix;
    Matrix identityMatrix;
    int size;

    inputMatrix = readMatrixFromFile();
    identityMatrix = createIdentityMatrix(inputMatrix.col);
    size = inputMatrix.col;

    bool invertible = true;
    
    #pragma omp barrier
    double start = omp_get_wtime();

    for (size_t i = 0; i < size; i++){
        /* Partial Pivoting */
        /* Swapping indivisible row */
        double* colBuffer = getColFromMatrix(inputMatrix, i);
        int swapIdx = -1;

        if (colBuffer[i] == 0. && invertible) {
            // Search for the nearest non-zero row
            for (size_t j = i + 1; j < size; j++) {
                if (colBuffer[j] != 0.) {
                    swapIdx = j;
                    break;
                } else if (j == size - 1) {
                    invertible = false;
                    fprintf(stderr, "Matrix cannot be inversed.\n");
                }
            }
        }

        if (swapIdx != -1) {
            swapRow(&inputMatrix, i, swapIdx);
            swapRow(&identityMatrix, i, swapIdx);
        }

        free(colBuffer);

        if (!invertible) {
            exit(1);
        }

        /* Eliminating */
        /* Reducing to upper triangle matrix */
        double eliminateFactor;

        #pragma omp for schedule(dynamic, 1)
        for (size_t j = 0; j < size; j++){
            if (i != j){
                eliminateFactor = inputMatrix.buffer[j * size + i] / inputMatrix.buffer[i * size + i];
                for (size_t k = 0; k < size; k++){
                    inputMatrix.buffer[j * size + k] -= inputMatrix.buffer[i * size + k] * eliminateFactor;
                    identityMatrix.buffer[j * size + k] -= identityMatrix.buffer[i * size + k] * eliminateFactor;
                }
            }
        }
        /* Divide the row by the pivot factor */
        double pivotFactor = inputMatrix.buffer[i * size + i];

        #pragma omp for schedule(dynamic, 1)
        for (int j = 0; j < size; j++){
            inputMatrix.buffer[i * size + j] /= pivotFactor;
            identityMatrix.buffer[i * size + j] /= pivotFactor;
        }
    }

    printf("%d\n",size);
    printMatrix(identityMatrix);

    #pragma omp barrier
    double end = omp_get_wtime();
    printf("Time taken is %.6f\n", end-start);

    return 0;
}