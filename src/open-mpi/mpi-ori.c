// mpicc src/matrix.c src/open-mpi/mpi.c -o src/open-mpi/mpi.out

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "../matrix.h"

void printRow (double* row, int column){
    printf("---------------PIVOT ROW------------------\n");
    for (int i=0; i<column; i++){
        printf("%.2f ", row[i]);
    }
    printf("\n---------------------------------");
}
void swapRow(Matrix* matrix, int row1, int row2){
    if(row1 < 0 || row1 >= matrix->row || row2 < 0 || row2 >= matrix->row){
        printf("Invalid row indices\n");
        return;
    }

    int startIndexRow2 = row2 * matrix->col;
    int startIndexRow1 = row1 * matrix->col;

    double* temp_row = (double*)malloc(matrix->col * sizeof(double));
    if(temp_row == NULL){
        printf("Memory allocation failed\n");
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

void swapRowDifferentMatrix(Matrix* matrix, int rowIdx, double* rowArr){
    // row idx validation
    if (rowIdx < 0 || rowIdx > matrix->row){
        printf("Invalid row indices\n");
        return;
    }

    // start idx on matrix
    int startIndexRow = rowIdx * matrix->row;

    // temporary row allocation
    double* temp_row = (double*)malloc(matrix->col * sizeof(double));
    if(temp_row == NULL){
        printf("Memory allocation failed\n");
        return;
    }

    //swap
    memcpy(temp_row, &(matrix->buffer[startIndexRow]), matrix->col * sizeof(double));
    memcpy(&(matrix->buffer[startIndexRow]), &(rowArr), matrix->col * sizeof(double));
    memcpy(&(rowArr), temp_row, matrix->col * sizeof(double));

    free(temp_row);
}

int main(void) {
    Matrix inputMatrix;
    Matrix identityMatrix;
    Matrix procInputMatrix;
    Matrix procOutputMatrix;
    int inputMatrixCol;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (world_rank == 0) {
        /* Init matrix */
        char filename[256];
        scanf("%255s", filename);

        inputMatrix = readMatrixFromFile(filename);
        printMatrix(inputMatrix);
        printf("-------\n");

        identityMatrix = createIdentityMatrix(inputMatrix.col);
        printMatrix(identityMatrix);
        printf("-------\n");

        MPI_Bcast(&inputMatrix.col, 1, MPI_INT, 0, MPI_COMM_WORLD);
        inputMatrixCol = inputMatrix.col;
    } else {
        MPI_Bcast(&inputMatrixCol, 1, MPI_INT, 0, MPI_COMM_WORLD);

    }

    /* Allocate an array for pivot row */
    double* pivotRowInput = (double*)malloc(inputMatrixCol * sizeof(double));
    double* pivotRowOutput = (double*)malloc(inputMatrixCol * sizeof(double));

    /* Define number of rows, start row & end row for each processes */
    int nRow = inputMatrixCol / world_size;
    int startRow = world_rank * nRow;
    int endRow = startRow + nRow;


    MPI_Request requestsInput[world_size];
    MPI_Request requestsOutput[world_size];
   
    /* Allocate input matrix & identity matrix in each processes */
    procInputMatrix = createMatrix(nRow, inputMatrixCol);
    procOutputMatrix = createMatrix(nRow, inputMatrixCol);
    // printf("inputMatrixCol process %d = %d\n", world_rank, inputMatrixCol);
    // printf("input matrix size process %d = (%d x %d)\n", world_rank, procInputMatrix.row, procInputMatrix.col);

    /* Scatter the input matrix & identity matrix from process 0 */
    MPI_Scatter(
        inputMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        procInputMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        0, MPI_COMM_WORLD);
    MPI_Scatter(
        identityMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        procOutputMatrix.buffer, procInputMatrix.row * procInputMatrix.col, MPI_DOUBLE, 
        0, MPI_COMM_WORLD);

    // printf("procInputMatrix from proc%d\n", world_rank);
    // printMatrix(procInputMatrix);
    // printMatrix(procOutputMatrix);

    /* Gauss Elimination */
    for (int row = 0; row < endRow; row++){
        int currRank = row / nRow;

        // printf("world_rank = %d; currRank = %d\n", world_rank, currRank);
        
        /* if the current rank is in this process */
        if (world_rank == currRank){
            int procRow = row % nRow;

            /* Partial pivoting */
            int pivotIdx = procRow * procInputMatrix.col + row;
            double pivot = procInputMatrix.buffer[pivotIdx];

            if (pivot == 0.){
                // printf("pivotnya 0 bang ga bisa -proc%d\n", world_rank);
                // swap row
                bool found = false;
                int rowNum = procRow + 1;

                while (!found){
                    // check if there's any row below in the same process and if the element below the pivot is not zero
                    if (rowNum < procInputMatrix.col 
                        && procInputMatrix.buffer[pivotIdx + procInputMatrix.col] != 0.) {
                        found = true;
                        // printf("i=%zu, rowNum=%zu\n", i, rowNum);
                        swapRow(&procInputMatrix, procRow, rowNum);
                        printMatrix(procInputMatrix);
                        pivot = procInputMatrix.buffer[procRow * procInputMatrix.col + row];
                        break;
                        // printMatrix(procInputMatrix);
                    } else if (world_rank < world_size-1) {
                        // check for other processes

                    } else {
                        printf("Matrix can not be inversed\n");
                        exit(1);
                    }
                }
            }

            for (int col = row; col < procInputMatrix.col; col++) {
                procInputMatrix.buffer[procRow * procInputMatrix.col + col] /= pivot;
                procOutputMatrix.buffer[procRow * procOutputMatrix.col + col] /= pivot;
            }
            // printf("\nPROCESS %d pivoting\nInput Matrix:\n", world_rank);
            // printMatrix(procInputMatrix);
            // printf("Output Matrix:\n");
            // printMatrix(procOutputMatrix);

            /* Send the pivot row to other processes */
            // MPI_Request send_req[2];
            for (int i = currRank + 1; i < world_size; i++) {
                MPI_Isend(procInputMatrix.buffer + procInputMatrix.col * procRow, 
                        procInputMatrix.col, MPI_DOUBLE, i, 0,
                        MPI_COMM_WORLD, &requestsInput[i]);
                MPI_Isend(procOutputMatrix.buffer + procOutputMatrix.col * procRow, 
                        procOutputMatrix.col, MPI_DOUBLE, i, 0,
                        MPI_COMM_WORLD, &requestsOutput[i]);
            }
            // MPI_Bcast(
            //     procInputMatrix.buffer + procRow * inputMatrixCol, inputMatrixCol, MPI_DOUBLE, 
            //     currRank, MPI_COMM_WORLD);
            // MPI_Bcast(
            //     procOutputMatrix.buffer + procRow * inputMatrixCol, inputMatrixCol, MPI_DOUBLE, 
            //     currRank, MPI_COMM_WORLD);

            /* Eliminate (zero-ing) the elements "below" the pivot */
            for (int eliminateRow = procRow + 1; eliminateRow < nRow; eliminateRow++) {
                // Get the scaling factor for elimination
                double factor = procInputMatrix.buffer[eliminateRow * procInputMatrix.col + row];

                // Execute subtraction
                for (int col = row; col < procInputMatrix.col; col++) {
                    procInputMatrix.buffer[eliminateRow * procInputMatrix.col + col] -= 
                        procInputMatrix.buffer[procRow * procInputMatrix.col + col] * factor;
                    procOutputMatrix.buffer[eliminateRow * procOutputMatrix.col + col] -= 
                        procOutputMatrix.buffer[procRow * procOutputMatrix.col + col] * factor;
                }
            }

            // Wait and check if there are any messages
            // MPI_Waitall(world_size, send_req, MPI_STATUSES_IGNORE);
            for (int i = currRank + 1; i < world_size-1; i++) {
                // printf("tunggu bnggg -proc%d\n", world_rank);
                MPI_Wait(&requestsInput[i], MPI_STATUS_IGNORE);
                MPI_Wait(&requestsOutput[i], MPI_STATUS_IGNORE);
                // MPI_Wait(&send_req[1], MPI_STATUS_IGNORE);
            }
        } else {
            /* other processes */
            /* Receive the pivot row from the "current rank process" */
            // printf("terima dulu -proc%d\n", world_rank);
            MPI_Recv(pivotRowInput, procInputMatrix.col, MPI_DOUBLE, currRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pivotRowOutput, procOutputMatrix.col, MPI_DOUBLE, currRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Skip rows that have been fully processed
            for (int eliminateRow = 0; eliminateRow < nRow; eliminateRow++) {
                // Get the scaling factor for elimination
                double factor = procInputMatrix.buffer[eliminateRow * procInputMatrix.col + row];

                // Remove the pivot
                for (int col = row; col < procInputMatrix.col; col++) {
                    procInputMatrix.buffer[eliminateRow * procInputMatrix.col + col] -= pivotRowInput[col] * factor;
                    procOutputMatrix.buffer[eliminateRow * procOutputMatrix.col + col] -= pivotRowOutput[col] * factor;
                }
            }

            // printf("\nPROCESS %d eliminating\nInput Matrix:\n", world_rank);
            // printMatrix(procInputMatrix);
            // printf("Output Matrix:\n");
            // printMatrix(procOutputMatrix);
        }
    }

    // printf("INPUT MATRIX proc%d:\n", world_rank);
    // printMatrix(procInputMatrix);
    // printf("OUTPUT MATRIX proc%d:\n", world_rank);
    // printMatrix(procOutputMatrix);

    /* Back Substitution */
    printf("=====Back Substitution=====\n");
    printf("endrow = %d, nRow = %d\n", endRow, nRow);
    for (int row = endRow - 1; row >= 0; row--){
        int currRank = row / nRow;

        // printf("world_rank = %d, row = %zu, currRank = %d\n", world_rank, row, currRank);

        /* if the current rank is in this process */
        if (world_rank == currRank){
            int procRow = row % nRow;

            /* Eliminate (zero-ing) the elements "above" the pivot */
            for (int eliminateRow = procRow - 1; eliminateRow >= 0; eliminateRow--) {
                // Get the scaling factor for elimination
                double factor = procInputMatrix.buffer[eliminateRow * procInputMatrix.col + row];

                // Execute subtraction
                for (int col = procInputMatrix.col - 1; col >= 0; col--) {
                    procInputMatrix.buffer[eliminateRow * procInputMatrix.col + col] -= 
                        procInputMatrix.buffer[procRow * procInputMatrix.col + col] * factor;
                    procOutputMatrix.buffer[eliminateRow * procOutputMatrix.col + col] -= 
                        procOutputMatrix.buffer[procRow * procOutputMatrix.col + col] * factor;
                }
                // for (int col = 0; col < procInputMatrix.col; col++) {
                //     procInputMatrix.buffer[eliminateRow * procInputMatrix.col + col] -= 
                //         procInputMatrix.buffer[procRow * procInputMatrix.col + col] * factor;
                //     procOutputMatrix.buffer[eliminateRow * procOutputMatrix.col + col] -= 
                //         procOutputMatrix.buffer[procRow * procOutputMatrix.col + col] * factor;
                // }
            }
            
            /* Send the pivot row to other processes */
            // MPI_Request send_req[2];
            printf("world_rank = %d, currRank = %d\n", world_rank, currRank);
            if (world_rank > 0) {
                for (int i = currRank - 1; i >= 0; i--) {
                    printf("kirim dari proc%d\n", world_rank);
                    MPI_Isend(procInputMatrix.buffer + procInputMatrix.col * procRow, 
                            procInputMatrix.col, MPI_DOUBLE, i, 0,
                            MPI_COMM_WORLD, &requestsInput[i]);
                    MPI_Isend(procOutputMatrix.buffer + procOutputMatrix.col * procRow, 
                            procOutputMatrix.col, MPI_DOUBLE, i, 0,
                            MPI_COMM_WORLD, &requestsOutput[i]);
                }
            }
            
            printf("\nPROCESS %d eliminating\nInput Matrix:\n", world_rank);
            printMatrix(procInputMatrix);
            printf("Output Matrix:\n");
            printMatrix(procOutputMatrix);

            // Wait and check if there are any messages
            // MPI_Waitall(world_size, send_req, MPI_STATUSES_IGNORE);
            // for (int i = currRank - 1; i >= 0; i--) {
            // if (world_rank < world_size - 1){
            //     printf("tunggu bnggg -proc%d\n", world_rank);
            //     MPI_Wait(&requestsInput[world_rank], MPI_STATUS_IGNORE);
            //     MPI_Wait(&requestsOutput[world_rank], MPI_STATUS_IGNORE);
                // MPI_Wait(&send_req[1], MPI_STATUS_IGNORE);
            // }
            if (world_rank < currRank){
                for (int i = currRank - 1; i  >= 0; i--) {
                    printf("tunggu bnggg -proc%d\n", world_rank);
                    MPI_Wait(&requestsInput[i], MPI_STATUS_IGNORE);
                    MPI_Wait(&requestsOutput[i], MPI_STATUS_IGNORE);
                    // MPI_Wait(&send_req[1], MPI_STATUS_IGNORE);
                }
            }
            // if (world_rank < world_size - 1) {
            //     printf("tunggu bnggg -proc%d\n", world_rank);
            //     MPI_Recv(pivotRowInput, procInputMatrix.col, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //     MPI_Recv(pivotRowOutput, procOutputMatrix.col, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // }
        } else {
            /* other processes */
            /* Receive the pivot row from the "current rank process" */
            // if (world_rank < currRank){
            for (int i = world_rank + 1; i < world_size; i++){
                printf("terima dulu -proc%d dari proc%d (row = %d)\n", world_rank, i, row);
                MPI_Recv(pivotRowInput, procInputMatrix.col, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(pivotRowOutput, procOutputMatrix.col, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // }

            printf("pivotrow from proc%d\n", world_rank);
            printRow(pivotRowInput,procInputMatrix.col);

            // Skip rows that have been fully processed
            for (int eliminateRow = nRow - 1; eliminateRow >= 0; eliminateRow--) {
                // Get the scaling factor for elimination
                double factor = procInputMatrix.buffer[eliminateRow * procInputMatrix.col + row];
                printf("factor: %.2f\n", factor);

                // Remove the pivot
                // for (int col = 0; col < procInputMatrix.col; col++){
                //     procInputMatrix.buffer[eliminateRow * procInputMatrix.col + col] -= pivotRowInput[col] * factor;
                //     procOutputMatrix.buffer[eliminateRow * procOutputMatrix.col + col] -= pivotRowOutput[col] * factor;
                // }
                for (int col = procInputMatrix.col - 1; col >= 0; col--) {
                    procInputMatrix.buffer[eliminateRow * procInputMatrix.col + col] -= pivotRowInput[col] * factor;
                    procOutputMatrix.buffer[eliminateRow * procOutputMatrix.col + col] -= pivotRowOutput[col] * factor;
                }
            }

            printf("\nPROCESS %d eliminating\nInput Matrix:\n", world_rank);
            printMatrix(procInputMatrix);
            printf("Output Matrix:\n");
            printMatrix(procOutputMatrix);
        }
    }





    /* Partial pivoting */
    /*
    for (int i = 0; i < procInputMatrix.row; i++){
        int pivotIdx = i * procInputMatrix.col + world_rank * procInputMatrix.row + i;

        // if matrix[i,i] == 0
        if (procInputMatrix.buffer[pivotIdx] == 0.){
            // swap row
            bool found = false;
            int rowNum = i+1;

            while (!found){
                // check if there's any row below in the same process and if the element below the pivot is not zero
                if (rowNum < procInputMatrix.col 
                    && procInputMatrix.buffer[pivotIdx + procInputMatrix.col] != 0.) {
                    found = true;
                    // printf("i=%zu, rowNum=%zu\n", i, rowNum);
                    swapRow(&procInputMatrix, i, rowNum);
                    // printMatrix(procInputMatrix);
                } else if (world_rank < world_size-1) {
                    // check for other processes, iterate other processes
                    
                    // int rank = world_rank+1;
                    // while (!found){
                    //     printf("process %d", world_rank);
                    //     if (world_rank == rank){
                    //         for (int procRow = 0; procRow < procInputMatrix.row; procRow++){
                    //             // check every elements below the pivotIdx
                    //             int idxCheck = pivotIdx - (procInputMatrix.row - procRow - 1) * procInputMatrix.col;
                    //             printf("idxCheck for process %d = %zu\n",world_rank, idxCheck);
                    //             if (procInputMatrix.buffer[idxCheck] != 0){
                    //                 found = true;
                    //                 // double* rowToBeSent = (double*)malloc(matrix->col * sizeof(double));
                    //                 // memcpy(
                    //                 //     rowToBeSent, 
                    //                 //     &(procInputMatrix->buffer[idxCheck - procInputMatrix.row*rank + 1]), 
                    //                 //     procInputMatrix.col * sizeof(double));

                    //                 // MPI_Send( const void* buf , MPI_Count count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm);
                    //                 break;
                    //             }
                    //         }
                    //     } else {

                    //     }
                    //     rank++;
                    // }

                } else {
                    printf("Matrix can not be inversed.\n");
                    exit(1);
                }
                rowNum++;
            }

        } else if (procInputMatrix.buffer[pivotIdx] != 1.) {
            int pivot = procInputMatrix.buffer[pivotIdx];
            for (int j = pivotIdx; j < pivotIdx + procInputMatrix.col - i - world_rank * procInputMatrix.row; j++) {
                procInputMatrix.buffer[j] /= pivot;
            }
        }
        printMatrix(procInputMatrix);
        printf("--------------\n");
    }
    */

    printf("dh beres bng -proc%d\n", world_rank);

    // Gather the final results into rank 0
    MPI_Gather(procInputMatrix.buffer, nRow * procInputMatrix.col, MPI_DOUBLE, inputMatrix.buffer, nRow * procInputMatrix.col, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(procOutputMatrix.buffer, nRow * procInputMatrix.col, MPI_DOUBLE, identityMatrix.buffer, nRow * procInputMatrix.col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    printf("Input Matrix:\n");
    printMatrix(inputMatrix);
    printf("Inversed Matrix:\n");
    printMatrix(identityMatrix);

    freeMatrix(&procInputMatrix);
    freeMatrix(&procOutputMatrix);

    if (world_rank == 0) {
        freeMatrix(&inputMatrix);
        freeMatrix(&identityMatrix);
    }

    return 0;
}