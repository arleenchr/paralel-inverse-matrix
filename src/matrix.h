#ifndef MATRIX_H
#define MATRIX_H

#define MAX_SIZE 512

struct Matrix {
    int size;
    double buffer[MAX_SIZE][MAX_SIZE];
};

void readMatrixFromFile(const char* filename, struct Matrix* matrix);
void printMatrix(struct Matrix* matrix);

#endif
