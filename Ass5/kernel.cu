
#include "common.h"

#include "timer.h"

#define COARSENING_FACTOR 16

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int bins_s[NUM_BINS];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM_BINS){
        bins_s[i] = 0;
    }
    __syncthreads();

    //Loading pixels and incementing the bins
    if (i < width * height){
        atomicAdd(
            &bins_s[image[i]], //Each thread is loading one pixel here
            1);
    }
    __syncthreads();

     //Commit the non-zero bin counts to the global copy of the histogram in parallel
    if (threadIdx.x < NUM_BINS) {
        if (bins_s[threadIdx.x] > 0)  
            atomicAdd(&bins[threadIdx.x], bins_s[threadIdx.x]);  
    }

}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    int numThreadsPerBlock = 256;
    int numBlocks = (width*height + numThreadsPerBlock - 1)/(numThreadsPerBlock);
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
}


__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int bins_s[NUM_BINS];

    //initialize bin_s to 0s
    int i = threadIdx.x + blockIdx.x * blockDim.x * COARSENING_FACTOR;
    
    if (threadIdx.x < NUM_BINS){
        bins_s[threadIdx.x] = 0;
    }
    __syncthreads();

    //Each thread load multiple pixels based on the COARSENING_FACTOR
    if (i < width * height){   
        for (int k = 0; k < COARSENING_FACTOR; ++k){
            if (i + k*blockDim.x < width * height) {
                atomicAdd(
                    &bins_s[image[i + k*blockDim.x]], //Each thread load its current pixel up to the COARSENING_FACTOR
                    1);
            }
        }
    }
    __syncthreads();
    //

    //Commit the non-zero bin counts to the global copy of the histogram in parallel
    if (threadIdx.x < NUM_BINS) {
        if (bins_s[threadIdx.x] > 0)  
            atomicAdd(&bins[threadIdx.x], bins_s[threadIdx.x]);  
    }

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    //Launch the grid (Note: the image has already been copied to global memory

    //Set the number of threads per block
    int numThreadsPerBlock = 256;
    int numBlocks = (width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);

}


