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

int main(void) {
    Matrix inputMatrix;
    Matrix identityMatrix;
    int size;

    inputMatrix = readMatrixFromFile();
    identityMatrix = createIdentityMatrix(inputMatrix.col);
    size = inputMatrix.col;
    bool invertible = true;

    
    printf("Input matrix:\n");
    for (size_t i=0; i<size; i++){
        for (size_t j=0; j<size; j++){
            printf("%.4f\t", inputMatrix.buffer[i * size + j]);
        }
        printf("\n");
    }
    printf("Identity matrix:\n");
    for (size_t i=0; i<size; i++){
        for (size_t j=0; j<size; j++){
            printf("%.4f\t", identityMatrix.buffer[i * size + j]);
        }
        printf("\n");
    }

    dim3 block(256,1,1);
    dim3 grid((size + block.x - 1) / block.x, 1, 1);

    // GPU allocation memory
    double *d_inputMatrix;
    double *d_identityMatrix;
    bool *d_invertible;
    int *d_size;

    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("Available device memory: %zu bytes\n", free_bytes);
    printf("Total device memory: %zu bytes\n", total_bytes);

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

    cudaFree(d_inputMatrix);
    cudaFree(d_identityMatrix);
    cudaFree(d_invertible);
    cudaFree(d_size);

    return 0;
}