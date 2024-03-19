#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void readMatrixFromFile(const char* filename, struct Matrix* matrix) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "../../test_cases/%s", filename);

    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    // Read matrix size
    if (fscanf(file, "%d", &matrix->size) != 1) {
        printf("Error reading matrix size.\n");
        exit(1);
    }

    // Size validation
    if (matrix->size <= 0 || matrix->size > MAX_SIZE) {
        printf("Invalid matrix size.\n");
        exit(1);
    }

    // Read matrix buffer
    for (int i = 0; i < matrix->size; i++) {
        for (int j = 0; j < matrix->size; j++) {
            if (fscanf(file, "%lf", &(matrix->buffer[i][j])) != 1) {
                printf("Error reading file.\n");
                exit(1);
            }
        }
    }

    fclose(file);
}

void printMatrix(struct Matrix* matrix){
    for (int i=0; i<matrix->size; i++){
        for (int j=0; j<matrix->size; j++){
            printf("%.16f ", matrix->buffer[i][j]);
        }
        printf("\n");
    }
}
