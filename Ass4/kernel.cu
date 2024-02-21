
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    //Bring the input tile to shared memory
    _shared_ float input_tile[IN_TILE_DIM][IN_TILE_DIM];
    int row = threadIdx.y + blockIdx.y * IN_TILE_DIM;
    int col = threadIdx.x + blockIdx.x * IN_TILE_DIM;

    if (row >= 0 && row < height && col >= 0 && col < width) {
        input_tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    //Compute the output tile
    _shared_ float output_tile[OUT_TILE_DIM][OUT_TILE_DIM];
    if (threadIdx.y < OUT_TILE_DIM && threadIdx.x < OUT_TILE_DIM) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_DIM; i++) {
            for (int j = 0; j < FILTER_DIM; j++) {
                sum += filter_c[i][j] * input_tile[i + threadIdx.y][j + threadIdx.x];
            }
        }
        output_tile[threadIdx.y][threadIdx.x] = sum;
    }
    __syncthreads();



}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM*FILTER_DIM*sizeof(float));
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {
    
    /*enough threads per block to load an entire input tile,
      enough blocks in the grid to process every output tile */

    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + IN_TILE_DIM - 1)/IN_TILE_DIM, (height + IN_TILE_DIM - 1)/IN_TILE_DIM);
    convolution_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);
    cudaDeviceSynchronize(); 
}

