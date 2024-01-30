
#include "common.h"

#include "timer.h"

__host__ __device__ double f(double a, double b) {
    return (a > b)? a : b;
}

__global__ void vecMax_kernel(double* a, double* b, double* c, unsigned int M) {
    // TODO
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < M) {
        c[i] = f(a[i], b[i]);
    }
    //
}

void vecMax_gpu(double* a, double* b, double* c, unsigned int M) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    double* a_d, b_d, c_d;
    cudaMalloc( (void**) &a_d , M*sizeof(double));
    cudaMalloc( (void**) &b_d , M*sizeof(double));
    cudaMalloc( (void**) &c_d , M*sizeof(double));
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemCopy(a_d, a, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemCopy(b_d, b, M*sizeof(double), cudaMemcpyHostToDevice);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (M + numThreadsPerBlock - 1)/numThreadsPerBlock;
    vecMax_kernel <<< numBlocks, numThreadsPerBlock >>> (&a_d, &b_d, &c_d, M);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(c, c_d, M*sizeof(double), cudaMemcpyDeviceToHost);
    //

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    //


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

