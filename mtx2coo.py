from scipy.io import mmread
import numpy as np
from scipy import sparse
from scipy.sparse import random

def gen_random_sparse_matrix(size, density):
    return random(size, size, density=density, \
        random_state=int(100000.0 * np.random.rand()))
# A = mmread('grid2.mtx')
# Coo = sparse.coo_matrix(A.toarray())
# B = A.toarray()
B = gen_random_sparse_matrix(5000, 0.3).toarray()
# print(B.shape)

print('''
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <fstream>
#include <time.h>

#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s", s);                              \
    if (yHostPtr)           free(yHostPtr);          \
    if (zHostPtr)           free(zHostPtr);          \
    if (xIndHostPtr)        free(xIndHostPtr);       \
    if (xValHostPtr)        free(xValHostPtr);       \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
    if (y)                  cudaFree(y);             \
    if (z)                  cudaFree(z);             \
    if (xInd)               cudaFree(xInd);          \
    if (xVal)               cudaFree(xVal);          \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
    if (cooRowIndex)        cudaFree(cooRowIndex);   \
    if (cooColIndex)        cudaFree(cooColIndex);   \
    if (cooVal)             cudaFree(cooVal);        \
    if (descr)              cusparseDestroyMatDescr(descr);\
    if (handle)             cusparseDestroy(handle); \
    cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)

int main(){
    cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int *    cooRowIndexHostPtr=0;
    int *    cooColIndexHostPtr=0;
    double * cooValHostPtr=0;
    int *    cooRowIndex=0;
    int *    cooColIndex=0;
    double * cooVal=0;
    int *    xIndHostPtr=0;
    double * xValHostPtr=0;
    double * yHostPtr=0;
    double * y_static=0;
    int *    xInd=0;
    double * xVal=0;
    double * y=0;
    int *    csrRowPtr=0;
    double * zHostPtr=0;
    double * z=0;
    int      n, nnz;
    double dzero =0.0;
    double done = 1.0;

    printf("testing example");
    /* create the following sparse test matrix in COO format */
    /* |1.0     2.0 3.0     |
       |    4.0             |
       |5.0     6.0 7.0     |
       |    8.0     9.0     |
       |                10.0| */

    n = 5000;      // rank of the matrix
    nnz = n * n * 0.3;   // number of non-zero elements
    
    cooRowIndexHostPtr = (int *)   malloc(nnz*sizeof(cooRowIndexHostPtr[0]));
    cooColIndexHostPtr = (int *)   malloc(nnz*sizeof(cooColIndexHostPtr[0]));
    cooValHostPtr      = (double *)malloc(nnz*sizeof(cooValHostPtr[0]));
    if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)){
        CLEANUP("Host malloc failed (matrix)");
        return 1;
    }
''')




cnt = 0
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        if B[i][j] != 0:
            print('cooRowIndexHostPtr[' + str(cnt) +']=' + str(i) + '; cooColIndexHostPtr[' + str(cnt) + ']=' + str(j) + '; cooValHostPtr[' + str(cnt) + ']=' + str(B[i][j]) + ';')
            cnt += 1



print('''
    /*
    //print the matrix
    printf("Input data:");
    for (int i=0; i<nnz; i++){
        printf("cooRowIndexHostPtr[%d]=%d  ",i,cooRowIndexHostPtr[i]);
        printf("cooColIndexHostPtr[%d]=%d  ",i,cooColIndexHostPtr[i]);
        printf("cooValHostPtr[%d]=%f     ",i,cooValHostPtr[i]);
    }
    */

    /* create a dense vector */
    /*  y  = [1.0 2.0 3.0 4.0 5.0] (dense) */
    yHostPtr    = (double *)malloc(n       *sizeof(yHostPtr[0]));
    y_static    = (double *)malloc(n       *sizeof(yHostPtr[0]));
    if(!yHostPtr || !y_static){
        CLEANUP("Host malloc failed (vectors)");
        return 1;
    }

    srand (time(NULL));
    for(int i = 0; i < n; i++){
        y_static[i] = rand() / double(RAND_MAX);
    }

    /*
    //print the vectors
    for (int j=0; j<1; j++){
        for (int i=0; i<n; i++){
            printf("yHostPtr[%d,%d]=%f",i,j,yHostPtr[i+n*j]);
        }
    }
    */

    /* allocate GPU memory and copy the matrix and vectors into it */
    cudaStat1 = cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0]));
    cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
    cudaStat3 = cudaMalloc((void**)&cooVal,     nnz*sizeof(cooVal[0]));
    cudaStat4 = cudaMalloc((void**)&y,          n*sizeof(y[0]));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess)) {
        CLEANUP("Device malloc failed");
        return 1;
    }
    cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,
                           (size_t)(nnz*sizeof(cooRowIndex[0])),
                           cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr,
                           (size_t)(nnz*sizeof(cooColIndex[0])),
                           cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cooVal,      cooValHostPtr,
                           (size_t)(nnz*sizeof(cooVal[0])),
                           cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y,           y_static,
                           (size_t)(n*sizeof(y[0])),
                           cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess)) {
        CLEANUP("Memcpy from Host to Device failed");
        return 1;
    }

    /* initialize cusparse library */
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }

    /* create and setup matrix descriptor */
    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    /* exercise conversion routines (convert matrix from COO 2 CSR format) */
    cudaStat1 = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0]));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (csrRowPtr)");
        return 1;
    }
    status= cusparseXcoo2csr(handle,cooRowIndex,nnz,n,
                             csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Conversion from COO to CSR format failed");
        return 1;
    }
    //csrRowPtr = [0 3 4 7 9]

    int devId;
    cudaDeviceProp prop;
    cudaError_t cudaStat;
    cudaStat = cudaGetDevice(&devId);
    if (cudaSuccess != cudaStat){
        CLEANUP("cudaGetDevice failed");
        printf("Error: cudaStat %d, %s", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }
    cudaStat = cudaGetDeviceProperties( &prop, devId) ;
    if (cudaSuccess != cudaStat){
        CLEANUP("cudaGetDeviceProperties failed");
        printf("Error: cudaStat %d, %s", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }

    /* SpmV */
    std::ofstream myfile;
    myfile.open ("example.txt");   
    printf("SpMV elapsed time:");
    for(int i = 0; i < 10; i++){
        srand (time(NULL));
        for(int i = 0; i < n; i++){
            y_static[i] = rand() / double(RAND_MAX);
        }

        cudaMemcpy(y, y_static, (size_t)(n*sizeof(y[0])), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                            &done, descr, cooVal, csrRowPtr, cooColIndex,
                            y, &dzero, y);
        cudaEventRecord(stop);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            CLEANUP("Matrix-vector multiplication failed");
            return 1;
        }
        cudaMemcpy(yHostPtr, y, (size_t)(n*sizeof(y[0])), cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float milliseconds = -1;
        cudaEventElapsedTime(&milliseconds, start, stop); 
        myfile << 1000.0 * milliseconds << " ";
        cudaDeviceSynchronize();
    }
    myfile.close();

    /* destroy matrix descriptor */
    status = cusparseDestroyMatDescr(descr);
    descr = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor destruction failed");
        return 1;
    }

    /* destroy handle */
    status = cusparseDestroy(handle);
    handle = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library release of resources failed");
        return 1;
    }

    printf("SpMV results:");
    for (int j=0; j<1; j++){
        for (int i=0; i<n; i++){
            printf("yHostPtr[%d,%d]=%f",i,j,yHostPtr[i+n*j]);
        }
    }
    
    CLEANUP("example test PASSED");
    return 0;
}

''')
