#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
	__shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];
    int row = threadIdx.y + blockIdx.y * OUT_TILE_DIM -FILTER_RADIUS;
    int col = threadIdx.x + blockIdx.x * OUT_TILE_DIM- FILTER_RADIUS;
	if((row >=0) && (row< height ) && (col>=0) && (col < width ) ) {
        in_s[threadIdx.y][threadIdx.x]=input[row*width + col];
    }else{
        in_s[threadIdx.y][threadIdx.x]=0.0f;
    }
	__syncthreads();
    if(threadIdx.y>=FILTER_RADIUS && threadIdx.y<IN_TILE_DIM-FILTER_RADIUS && threadIdx.x>=FILTER_RADIUS && threadIdx.x< IN_TILE_DIM-FILTER_RADIUS){
		float sum = 0.0f;
        for(int i = 0; i < FILTER_DIM; i++) {
			for(int j = 0; j < FILTER_DIM; j++) { 
				sum += filter_c[i][j] * in_s[i+threadIdx.y-FILTER_RADIUS][j+threadIdx.x-FILTER_RADIUS];
            } 
        }
        if(row < height && col < width){
			output[row*width + col] = sum;
        }
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM*FILTER_DIM*sizeof(float));

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel

    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);
}