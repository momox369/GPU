
#include "common.h"

#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO
    // rows = M and cols = K
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < K && col < M){
        float sum = 0.0f;
        for(unsigned int j = 0; j < N; ++j) {
            sum += A[row*N + j]*B[j*N + col];
        }
        C[row*N + col] = sum;
    }

}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    float* A_d;
    float* B_d;
    float* C_d;
    cudaMalloc((void**)&A_d, sizeof(float)*M*N);
    cudaMalloc((void**)&B_d, sizeof(float)*N*K);
    cudaMalloc((void**)&C_d, sizeof(float)*M*K);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(A_d, A, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(float)*N*K, cudaMemcpyHostToDevice);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    // C is a MxK matrix
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((M + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (K + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    mm_kernel <<<numBlocks, numThreadsPerBlock>>> (A_d, B_d, C_d, M, N, K);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(C, C_d, M*K*sizeof(float), cudaMemcpyDeviceToHost);
    //


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

