
#include "common.h"

#include "timer.h"

#define COARSENING_FACTOR 4

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int* bins_s;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < width * height){
        bins_s[i] = 0;
    }
    __syncthreads();

    if (i < width * height){
        atomicAdd(
            &bins_s[image[i]], //Each thread is loading one pixel here
            1);
    }
    __syncthreads();

     //Commit the non-zero bin counts to the global copy of the histogram in parallel
    if (i < width * height) {
        atomicAdd(
                &bins[image[i]],
                bins_s[image[i]]);   //still counted as parallel?
    }

}

void histogram_gpu_private_(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    int numThreadsPerBlock = 256;
    int numBlocks = (width*height + numThreadsPerBlock - 1)/(numThreadsPerBlock + COARSENING_FACTOR);
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
}


__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int* bins_s;

    //initialize bin_s to 0s
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //second option: int i = blockIdx.x * blockDim.x * COARSENNING_FACTOR + threadIdx.x;
    
    if (i < width * height){
        bins_s[i] = 0;
    }
    __syncthreads();

    //Each thread load multiple pixels based on the COARSENING_FACTOR
    if ((i < width * height) && (i % COARSENING_FACTOR == 0)){   //ask Dr Izzat about it
        for (int k = 0; k < COARSENING_FACTOR; k++){
            if (i + k < width * height) {
                atomicAdd(
                    &bins_s[image[i + k]], //Each thread load its current pixel up to the COARSENING_FACTOR
                    1);
            }
        }
    }
    __syncthreads();
    //

    //Commit the non-zero bin counts to the global copy of the histogram in parallel
    if (i < width * height){
        atomicAdd(
         &bins[image[i]],
         bins_s[image[i]]);   //still counted as parallel?
    }

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    //Launch the grid (Note: the image has already been copied to global memory

    //Set the number of threads per block
    int numThreadsPerBlock = 512;
    int numBlocks = (width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);

}


