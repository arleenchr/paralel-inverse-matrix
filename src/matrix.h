#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int col;
    int row;
    double* buffer;
} Matrix;

Matrix createMatrix(int row, int col);
void freeMatrix(Matrix *mat);
Matrix readMatrixFromFile(const char* filename);
Matrix createIdentityMatrix(int size);
void printMatrix(Matrix matrix);

#endif