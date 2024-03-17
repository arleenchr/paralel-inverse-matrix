// mpicc mpi.c -o mpi

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 2048

struct Matrix {
    int size;
    double buffer[MAX_SIZE][MAX_SIZE];
};

void readMatrixFromInput(struct Matrix *matrix) {
    scanf("Matrix size = %d", &matrix->size);

    for (int i=0; i<matrix->size; i++){
        for (int j=0; j<matrix->size; j++){
            scanf("[%d][%d] = %lf", &i, &j, &matrix->buffer[i][j]);
        }
    }
}

void readMatrixFromFile(const char* filename, struct Matrix* matrix) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

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
            printf("%.4f ", matrix->buffer[i][j]);
        }
        printf("\n");
    }
}

int main(void) {
    // struct Matrix m;

    // readMatrixFromFile("../../test_cases/32.txt", &m);
    // readMatrixFromInput(&m);
    // printMatrix(&m);

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    struct Matrix inputMatrix;
    struct Matrix outputMatrix;

    printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    MPI_Finalize();

    return 0;
}