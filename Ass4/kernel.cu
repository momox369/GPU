
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    //Bring the input tile to shared memory
    __shared__ float input_tile[IN_TILE_DIM][IN_TILE_DIM];
    
    int in_row = (blockIdx.y * OUT_TILE_DIM) + threadIdx.y - FILTER_RADIUS; 
    int in_col = (blockIdx.x * OUT_TILE_DIM) + threadIdx.x - FILTER_RADIUS;

    if (in_row >= 0 && in_row < height  && in_col >= 0 && in_col < width ) {
        input_tile[threadIdx.y][threadIdx.x] = input[in_row*width + in_col];
    } 
    else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    //Compute filter * input_tile
    //if ((in_row >= FILTER_RADIUS && in_row < height - FILTER_RADIUS) & (in_col >= FILTER_RADIUS && in_row < width - FILTER_RADIUS)) { //boundary for computing inner tile
    if ((in_row >= FILTER_RADIUS && in_row < IN_TILE_DIM - FILTER_RADIUS) & (in_col >= FILTER_RADIUS && in_row < IN_TILE_DIM - FILTER_RADIUS)) { //boundary for computing inner tile
        if (threadIdx.y >= FILTER_RADIUS && threadIdx.y < OUT_TILE_DIM && threadIdx.x >= FILTER_RADIUS && threadIdx.x < OUT_TILE_DIM) {

            float sum = 0.0f;
            for(int out_row = 0; out_row < FILTER_DIM; out_row++) {
			    for(int out_col = 0; out_col < FILTER_DIM; out_col++) { 
				    sum += filter_c[out_row][out_col] * input_tile[out_row + threadIdx.y - FILTER_RADIUS][out_col + threadIdx.x - FILTER_RADIUS];
                } 
            }
       
            if (in_row < height && in_col < width) {
			    output[in_row*width + in_col] = sum;
            }
        }
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {

    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM*FILTER_DIM*sizeof(float));

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Configure and Call kernel
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
}

