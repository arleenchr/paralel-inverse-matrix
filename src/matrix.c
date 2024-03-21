#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

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

Matrix readMatrixFromFile(const char* filename){
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "test_cases/%s", filename);

    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    int size;

    // Read matrix size
    if (fscanf(file, "%d", &size) != 1) {
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
            if (fscanf(file, "%lf", &(matrix.buffer[i * matrix.col + j])) != 1) {
                printf("Error reading file.\n");
                exit(1);
            }
        }
    }

    fclose(file);
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
            printf("%.2f\t", matrix.buffer[i * matrix.col + j]);
        }
        printf("\n");
    }
}
