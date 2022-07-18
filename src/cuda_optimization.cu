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
__global__ void sep_row_conv(float *r, float *g, float *b, float *r_out, float *g_out, float *b_out, int width, int height, float *filter_kernel){
    
    extern __shared__ float data_r[];
    extern __shared__ float data_g[];
    extern __shared__ float data_b[];

    int radius = 1; // defined kernels only have radius of 1
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // sanity check
    if (x_idx >= width || y_idx >= height) return;

    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;

    float acc_r = 0;
    float acc_g = 0;
    float acc_b = 0;

    float val_r = 0;
    float val_g = 0;
    float val_b = 0;

    int w = blockDim.x;

#pragma unroll 3 
    for (int i = -w; i <= w; i += w){
        int x_zero = threadIdx.x + i;
        int new_loc = loc + i;
        if (x_zero < -radius || x_zero >= radius + w || new_loc < 0 || new_loc >= width*height)
            continue;

        data_r[threadIdx.x + i + radius + (threadIdx.y) * (blockDim.x + (radius << 1))] = r[new_loc];
        data_g[threadIdx.x + i + radius + (threadIdx.y) * (blockDim.x + (radius << 1))] = g[new_loc];
        data_b[threadIdx.x + i + radius + (threadIdx.y) * (blockDim.x + (radius << 1))] = b[new_loc];
    }

    __syncthreads();

    for (int i = -radius; i <= radius; i++){
        float t_r = data_r[threadIdx.x + i + radius + (threadIdx.y) * (blockDim.x + (radius << 1))];
        float t_g = data_g[threadIdx.x + i + radius + (threadIdx.y) * (blockDim.x + (radius << 1))];
        float t_b = data_b[threadIdx.x + i + radius + (threadIdx.y) * (blockDim.x + (radius << 1))];

        float temp = filter_kernel[i + radius];

        val_r = t_r;
        val_g = t_g;
        val_b = t_b;

        val_r *= temp;
        val_g *= temp;
        val_b *= temp;

        acc_r += val_r;
        acc_g += val_g;
        acc_b += val_b;
    }

    r_out[loc] = acc_r;
    g_out[loc] = acc_g;
    b_out[loc] = acc_b;
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
    sep_row_conv<<<num_blocks, threads_per_block>>>(d_R, d_G, d_B, d_R_out, d_G_out, d_B_out, r.cols, r.rows, H_RB);
    // conv_2d<<<num_blocks, threads_per_block>>>(d_G, d_G_out, g.cols, g.rows, g.step, radius, H_G);
    // conv_2d<<<num_blocks, threads_per_block>>>(d_B, d_B_out, b.cols, b.rows, b.step, radius, H_RB);

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