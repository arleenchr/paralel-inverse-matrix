#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "matrix.h"

Matrix createMatrix(int row, int col) {
    Matrix matrix;
    matrix.col = col;
    matrix.row = row;
    
    matrix.buffer = (double *)malloc(col * row * sizeof(double));
    if (matrix.buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    return matrix;
}

void freeMatrix(Matrix *matrix) {
    free(matrix->buffer);

    matrix->buffer = NULL;
    matrix->col = 0;
    matrix->row = 0;
}

Matrix readMatrixFromFile(){
    int size;

    // Read matrix size
    if (fscanf(stdin, "%d", &size) != 1) {
        printf("Error reading matrix size.\n");
        exit(1);
    }

    Matrix matrix = createMatrix(size, size);

    // Size validation
    if (matrix.col <= 0 || matrix.row <= 0 || matrix.col % 4 != 0 || matrix.row % 4 != 0) {
        printf("Invalid matrix size.\n");
        exit(1);
    }

    // Read matrix buffer
    for (int i = 0; i < matrix.row; i++) {
        for (int j = 0; j < matrix.col; j++) {
            if (fscanf(stdin, "%lf", &(matrix.buffer[i * matrix.col + j])) != 1) {
                printf("Error reading file.\n");
                exit(1);
            }
        }
    }
    return matrix;
}

Matrix createIdentityMatrix(int size){
    Matrix matrix = createMatrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                matrix.buffer[i * size + j] = 1.0;
            } else {
                matrix.buffer[i * size + j] = 0.0;
            }
        }
    }
    return matrix;
}

void printMatrix(Matrix matrix){
    for (int i = 0; i < matrix.row; i++) {
        for (int j = 0; j < matrix.col; j++) {
            printf("%.6f ", matrix.buffer[i * matrix.col + j]);
        }
        printf("\n");
    }
}

double* getColFromMatrix(Matrix m, size_t colNum){
    double* col = (double *)malloc(m.row * sizeof(double));
    if (col == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    for (size_t i = 0; i < m.row; i++){
        col[i] = m.buffer[i * m.col + colNum];
    }
    return col;
}