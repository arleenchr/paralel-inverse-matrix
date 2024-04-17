#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

extern "C"{
  #include "matrix.h"
};

#define CUDA_CHECK_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(err); \
    } \
} while(0)

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

__global__ void eliminate(double* inputMatrix, double* identityMatrix, int size, size_t it){
    size_t row = (blockIdx.x*blockDim.x) + threadIdx.x;
	size_t col = (blockIdx.y*blockDim.y) + threadIdx.y;

    if (row < size && col < size && row != it){
        identityMatrix[row * size + col] -= identityMatrix[it * size + col] * inputMatrix[row * size + it];
        
        if (col != it) {
            inputMatrix[row * size + col] -= inputMatrix[it * size + col] * inputMatrix[row * size + it];
        }
    }
}

__global__ void normalize(double* inputMatrix, double* identityMatrix, int size, size_t it){
    size_t row = (blockIdx.x*blockDim.x) + threadIdx.x;
	size_t col = (blockIdx.y*blockDim.y) + threadIdx.y;

    if (row < size && col < size && row != it && col == it) {
       inputMatrix[row * size + col] = 0;
    }
}

__global__ void reduce_nodiag(double* inputMatrix, double* identityMatrix, int size, size_t it){
    size_t row = (blockIdx.x*blockDim.x) + threadIdx.x;
	  size_t col = (blockIdx.y*blockDim.y) + threadIdx.y;
    if (row < size && col < size && row == it && row != col){
        double pivotFactor = inputMatrix[it * size + it];
        inputMatrix[it * size + col] /= pivotFactor;
        identityMatrix[it * size + col] /= pivotFactor;
    }
}

__global__ void reduce_diag(double* inputMatrix, double* identityMatrix, int size, size_t it){
    size_t row = (blockIdx.x*blockDim.x) + threadIdx.x;
	  size_t col = (blockIdx.y*blockDim.y) + threadIdx.y;
    if (row < size && col < size && row == col && row == it){
        double pivotFactor = inputMatrix[it * size + it];

        inputMatrix[it * size + col] /= pivotFactor;
        identityMatrix[it * size + col] /= pivotFactor;
    }
}

int main(void) {
    Matrix inputMatrix;
    Matrix identityMatrix;
    int size;
    
    inputMatrix = readMatrixFromFile();
    identityMatrix = createIdentityMatrix(inputMatrix.col);
    size = inputMatrix.col;
    bool invertible = true;

    dim3 block(16,16);
    int gridRow = (size+15)/16;
    int gridCol = (size+15)/16;
    dim3 grid(gridRow,gridCol);

    // GPU allocation memory
    double *d_inputMatrix;
    double *d_identityMatrix;
    bool *d_invertible;
    int *d_size;

    cudaMalloc((void **)&d_inputMatrix, size * size * sizeof(double));
    CUDA_CHECK_ERROR();

    cudaMalloc((void **)&d_identityMatrix, size * size * sizeof(double));
    CUDA_CHECK_ERROR();

    cudaMalloc((void **)&d_invertible, sizeof(bool));
    CUDA_CHECK_ERROR();

    cudaMalloc((void **)&d_size, sizeof(int));
    CUDA_CHECK_ERROR();

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
        free(colBuffer);

        cudaMemcpy(d_inputMatrix, inputMatrix.buffer, size * size * sizeof(double), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMemcpy(d_identityMatrix, identityMatrix.buffer, size * size * sizeof(double), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMemcpy(d_invertible, &invertible, sizeof(bool), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        reduce_nodiag<<<grid,block>>>(d_inputMatrix, d_identityMatrix, size, i);
        CUDA_CHECK_ERROR();

        reduce_diag<<<grid,block>>>(d_inputMatrix, d_identityMatrix, size, i);
        CUDA_CHECK_ERROR();

        eliminate<<<grid,block>>>(d_inputMatrix, d_identityMatrix, size, i);
        CUDA_CHECK_ERROR();

        normalize<<<grid,block>>>(d_inputMatrix, d_identityMatrix, size, i);
        CUDA_CHECK_ERROR();

        cudaDeviceSynchronize();

        cudaMemcpy(inputMatrix.buffer, d_inputMatrix, size * size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(identityMatrix.buffer, d_identityMatrix, size * size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&invertible, d_invertible, sizeof(bool), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
    }

    printf("%d\n", size);
    printMatrix(identityMatrix);

    cudaFree(d_inputMatrix);
    cudaFree(d_identityMatrix);
    cudaFree(d_invertible);
    cudaFree(d_size);

    freeMatrix(&inputMatrix);
    freeMatrix(&identityMatrix);

    return 0;
}