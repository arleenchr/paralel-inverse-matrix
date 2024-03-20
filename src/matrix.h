#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int size;
    double** buffer;
} Matrix;

Matrix createMatrix(int size);
void freeMatrix(Matrix *mat);
Matrix readMatrixFromFile(const char* filename);
Matrix createIdentityMatrix(int size);
void printMatrix(Matrix matrix);

#endif
