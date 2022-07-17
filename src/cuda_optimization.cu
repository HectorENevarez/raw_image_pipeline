#include <iostream>

// local include
#include "cuda_optimization.h"


// -----------------------------------------------------------------------------------------------------------------------------
// Defined kernels for convolutions
// -----------------------------------------------------------------------------------------------------------------------------
__constant__ float H_G[] =
{
    0.00, 0.25, 0.00,
    0.25, 1.00, 0.25,
    0.00, 0.25, 0.00,
};

__constant__ float H_RB[] =
{
    0.25, 0.50, 0.25,
    0.50, 1.00, 0.50,
    0.25, 0.50, 0.25,
};


// -----------------------------------------------------------------------------------------------------------------------------
// Better CUDA error handling check
// -----------------------------------------------------------------------------------------------------------------------------
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


// -----------------------------------------------------------------------------------------------------------------------------
// Kernel convolution logic
// -----------------------------------------------------------------------------------------------------------------------------
__global__ void conv_2d(float *channel, float *channel_out, int width, int height, int step, float *filter_kernel){
    int radius = 1; // defined kernels only have radius of 1
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("(%d %d)\n", x_idx, y_idx);

    // Make sure that we're within bounds
    if ((x_idx < width) && (y_idx < height)) {
        float val = 0;
        float acc_val = 0;

        // location of pixel
		int pix_id = y_idx * step + x_idx;

		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++) {

				//Skip violations (which will lead to zero by default
				if ((x_idx + i < 0) || (x_idx + i >= width) || (y_idx + j < 0) || (y_idx + j >= height)) continue;

				//Get kernel value
				int temp = filter_kernel[i + radius + (j+radius)*((radius << 1) + 1)];

				//Fetch the three channel values
                const float pix_val = channel[pix_id];
                val = pix_val * temp;
                
                // cumulative sum
                acc_val += val;
			}
		}

        channel_out[pix_id] = acc_val;
    }

}


// -----------------------------------------------------------------------------------------------------------------------------
// Performs 2D convolutions for demosaicing raw images
// -----------------------------------------------------------------------------------------------------------------------------
int convolution_2d_cuda(const cv::Mat& r, const cv::Mat& g, const cv::Mat& b, cv::Mat& demosaiced) {
    const int channel_bytes = r.step * r.rows;
    
    // allocate color channels in device memory
    float *d_R, *d_R_out;
    float *d_G, *d_G_out;
    float *d_B, *d_B_out;

    SAFE_CALL(cudaMalloc(&d_R, channel_bytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_G, channel_bytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_B, channel_bytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_R_out, channel_bytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_G_out, channel_bytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_B_out, channel_bytes), "CUDA Malloc Failed");

    // define block and grid size dynamically for different sized images
    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks((r.cols + threads_per_block.x - 1)/threads_per_block.x, \
                         (r.rows + threads_per_block.y - 1)/threads_per_block.y);

    // copy into device memory
    SAFE_CALL(cudaMemcpy(d_R, r.ptr(), channel_bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_G, g.ptr(), channel_bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_B, b.ptr(), channel_bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");


    // launch the convolution kernel
    conv_2d<<<num_blocks, threads_per_block>>>(d_R, d_R_out, r.cols, r.rows, r.step, H_RB);
    conv_2d<<<num_blocks, threads_per_block>>>(d_G, d_G_out, g.cols, g.rows, g.step, radius, H_G);
    conv_2d<<<num_blocks, threads_per_block>>>(d_B, d_B_out, b.cols, b.rows, b.step, radius, H_RB);

    // wait for threads to complete
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    // free resources
    SAFE_CALL(cudaFree(d_R),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_G),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_B),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_R_out),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_G_out),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_B_out),"CUDA Free Failed");


    return 0;
}