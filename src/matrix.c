#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

Matrix createMatrix(int size) {
    Matrix mat;
    mat.size = size;
    
    mat.buffer = (double **)malloc(size * sizeof(double *));
    if (mat.buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        mat.buffer[i] = (double *)malloc(size * sizeof(double));
        if (mat.buffer[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
    }
    
    return mat;
}

void freeMatrix(Matrix *mat) {
    for (int i = 0; i < mat->size; i++) {
        free(mat->buffer[i]);
    }
    free(mat->buffer);
}

Matrix readMatrixFromFile(const char* filename) {
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

    Matrix matrix = createMatrix(size);

    // Size validation
    if (matrix.size <= 0 || matrix.size % 4 != 0) {
        printf("Invalid matrix size.\n");
        exit(1);
    }

    // Read matrix buffer
    for (int i = 0; i < matrix.size; i++) {
        for (int j = 0; j < matrix.size; j++) {
            if (fscanf(file, "%lf", &(matrix.buffer[i][j])) != 1) {
                printf("Error reading file.\n");
                exit(1);
            }
        }
    }

    fclose(file);
    return matrix;
}

Matrix createIdentityMatrix(int size){
    Matrix matrix = createMatrix(size);
    for (int i=0; i<matrix.size; i++){
        for (int j=0; j<matrix.size; j++){
            if (i == j) {
                matrix.buffer[i][j] = 1.;
            } else {
                matrix.buffer[i][j] = 0.;
            }
        }
    }
    return matrix;
}

void printMatrix(Matrix matrix){
    for (int i=0; i<matrix.size; i++){
        for (int j=0; j<matrix.size; j++){
            printf("%.16f ", matrix.buffer[i][j]);
        }
        printf("\n");
    }
}
