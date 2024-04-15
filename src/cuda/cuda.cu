#include <time.h>
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

__device__ void swapRow(double *matrix, int row1, int row2, int *size){
    int startIndexRow1 = row1 * (*size);
    int startIndexRow2 = row2 * (*size);
    
    double* temp_row = (double*)malloc((*size) * sizeof(double));
    if(temp_row == NULL){
        printf("Memory allocation failed\n");
        return;
    }

    // Copy row1 to temp_row
    memcpy(temp_row, &(matrix[startIndexRow1]), (*size) * sizeof(double));

    // Copy row2 to row1
    memcpy(&(matrix[startIndexRow1]), &(matrix[startIndexRow2]), (*size) * sizeof(double));

    // Copy temp_row to row2
    memcpy(&(matrix[startIndexRow2]), temp_row, (*size) * sizeof(double));

    free(temp_row);
}

__device__ double* getColFromMatrixBuffer(double *m, int *size, int colNum){
    double* col = (double *)malloc((*size) * sizeof(double));

    for (int i = 0; i < *size; i++){
        col[i] = m[i * (*size) + colNum];
    }
    
    return col;
}

__global__ void pivoting(double *inputBuffer, double *identityBuffer, int *size, bool *invertible){
    for (int i = 0; i < *size; i++){
        double* colBuffer = getColFromMatrixBuffer(inputBuffer, size, i);
        
        if (colBuffer[i] == 0.){
            for (int swapIdx = i+1; swapIdx < *size; swapIdx++){
                if (colBuffer[swapIdx] != 0.){
                    // printf("SWAP\n");
                    swapRow(inputBuffer, i, swapIdx, size);
                    swapRow(identityBuffer, i, swapIdx, size);
                    break;
                } else if (swapIdx == *size - 1){
                    *invertible = false;
                    printf("Matrix can not be inversed.\n");
                }
            }
        }
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

    dim3 block(256,1,1);
    dim3 grid((size + block.x - 1) / block.x, 1, 1);

    // GPU allocation memory
    double *d_inputMatrix;
    double *d_identityMatrix;
    bool *d_invertible;
    int *d_size;

    // size_t free_bytes, total_bytes;
    // cudaMemGetInfo(&free_bytes, &total_bytes);
    // printf("Available device memory: %zu bytes\n", free_bytes);
    // printf("Total device memory: %zu bytes\n", total_bytes);

    cudaMalloc((void **)&d_inputMatrix, size * size * sizeof(double));
    CUDA_CHECK_ERROR();

    cudaMalloc((void **)&d_identityMatrix, size * size * sizeof(double));
    CUDA_CHECK_ERROR();

    cudaMalloc((void **)&d_invertible, sizeof(bool));
    CUDA_CHECK_ERROR();

    cudaMalloc((void **)&d_size, sizeof(int));
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_inputMatrix, inputMatrix.buffer, size * size * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_identityMatrix, identityMatrix.buffer, size * size * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_invertible, &invertible, sizeof(bool), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    /* Start clock */
    clock_t start = clock();

    /* Partial Pivoting */
    pivoting<<<grid,block>>>(d_inputMatrix, d_identityMatrix, d_size, d_invertible);
    cudaDeviceSynchronize();

    if (!d_invertible){
        return;
    }

    cudaMemcpy(inputMatrix.buffer, d_inputMatrix, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(identityMatrix.buffer, d_identityMatrix, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&invertible, d_invertible, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size, d_size, sizeof(int), cudaMemcpyDeviceToHost);

    
    printf("Input matrix:\n");
    double *inputMatrixHost = (double *)malloc(size * size * sizeof(double));
    cudaMemcpy(inputMatrixHost, d_inputMatrix, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%.4f\t", inputMatrixHost[i * size + j]);
        }
        printf("\n");
    }
    free(inputMatrixHost);

    printf("Identity matrix:\n");
    double *identityMatrixHost = (double *)malloc(size * size * sizeof(double));
    cudaMemcpy(identityMatrixHost, d_identityMatrix, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%.4f\t", identityMatrixHost[i * size + j]);
        }
        printf("\n");
    }
    free(identityMatrixHost);

    /* Stop clock */
    clock_t end = clock();
    double exectime = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Time taken is %.6f\n", exectime);

    cudaFree(d_inputMatrix);
    cudaFree(d_identityMatrix);
    cudaFree(d_invertible);
    cudaFree(d_size);

    return 0;
}