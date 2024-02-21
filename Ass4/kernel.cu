
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    //Bring the input tile to shared memory
    __shared__ float input_tile[IN_TILE_DIM][IN_TILE_DIM];
    int in_row = (blockIdx.y * OUT_TILE_DIM) + threadIdx.y; //should I add -1 here?
    int in_col = (blockIdx.x * OUT_TILE_DIM) + threadIdx.x;

    //loading
    if ((in_row >= 0) && (in_row < height ) && (in_col >= 0) && (in_col < width ) ) {
        input_tile[threadIdx.y][threadIdx.x] = input[in_row*width + in_col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    //Compute filter x input_tile
    float sum = 0.0f;
    for (int filter_row = 0; filter_row < FILTER_DIM; ++filter_row){

        for (int filter_col = 0; filter_col < FILTER_DIM; ++filter_col){

            int out_row = in_row + filter_row - FILTER_RADIUS;
            int out_col = in_col + filter_col - FILTER_RADIUS;

            if ((out_row >= 0) && (out_row < height ) && (out_col >= 0) && (out_col < width ) ) {
                sum += input_tile[threadIdx.y + filter_row - FILTER_RADIUS][threadIdx.x + filter_col - FILTER_RADIUS] * filter_c[filter_row][filter_col];
            }
        }
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

