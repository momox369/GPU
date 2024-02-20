
#include "common.h"

#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO
    //write the kernel for matrix multiplication using tiling
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float Cvalue = 0.0;

    for (int m = 0; m < (N + TILE_DIM - 1)/TILE_DIM; m++) {

        //loading data from gloabal memory to shared memory
        if (m * TILE_DIM + tx < N && row < M) {
            As[ty][tx] = A[row * N + m * TILE_DIM + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        if (m * TILE_DIM + ty < N && col < K) {
            Bs[ty][tx] = B[(m * TILE_DIM + ty) * K + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        __syncthreads();

        //computing the result
        for (int k = 0; k < TILE_DIM; k++) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = Cvalue;
    }


}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(d_A, A, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*N*K, cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((M + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (K + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    mm_tiled_kernel <<<numBlocks, numThreadsPerBlock>>> (d_A, d_B, d_C, M, N, K);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

