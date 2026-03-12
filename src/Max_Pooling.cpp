#include "Max_Pooling.hpp"
#include "Drive_Singleton.hpp"

Max_Pooling::Max_Pooling(int batch_size, int depth, int image_height, int image_width) : Layer()
{
    this->batch_size = batch_size;
    this->depth = depth;
    this->H = image_height;
    this->W = image_width;
    // this->H_out = H/2;
    // this->W_out = W/2;
    this->H_out = (H + 1) / 2;
    this->W_out = (W + 1) / 2;

    CUmodule mod = Driver_Singleton::getInstance()->getModule();
    CUresult res1 = cuModuleGetFunction(&k_max_pooling_forward, mod, "max_pooling_forward_kernel_2D");
    CUresult res2 = cuModuleGetFunction(&k_max_pooling_backward, mod, "max_pooling_backward_kernel_2D");
    
    if (res1 != CUDA_SUCCESS) {
        throw std::runtime_error("[Max_Pooling] Cannot find kernel 'max_pooling_forward_kernel' in PTX");
    }

    if (res2 != CUDA_SUCCESS) {
        throw std::runtime_error("[Max_Pooling] Cannot find kernel 'max_pooling_backward_kernel' in PTX");
    }

    this->total_size = this->batch_size * this->depth * H_out * W_out;
    this->in_size = this->batch_size * this->depth * H * W;
    cudaMalloc(&dY_position, (size_t)total_size * sizeof(int));
    
}

Max_Pooling::~Max_Pooling()
{
    if(this->dY_position)
    {
        cudaFree(dY_position);
        dY_position = nullptr;
    }
    if(this->d_output)
    {
        cudaFree(d_output);
        d_output = nullptr;
    }
    if(this->d_cache_con)
    {
        cudaFree(d_cache_con);
        d_cache_con = nullptr;
    }
}

float *Max_Pooling::forward(float *d_input, int batch_size)
{
    this->batch_size = batch_size;
    if(d_input == nullptr)
    {
        throw std::runtime_error("[Max_Pooling] Max_Pooling's input requires the precede module's output memory");
    }

    if(this->d_output == nullptr)
    {
        cudaMalloc(&d_output, (size_t)total_size* sizeof(float));
    }

    int BLOCK_SIZE = 16;
    int grid_w = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_h = (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_w * grid_h, depth, this->batch_size);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

    // int threads_per_block = 256;
    // int blocks_per_grid = (total_size + threads_per_block - 1) / threads_per_block ;
    // dim3 grid(blocks_per_grid, 1);
    // dim3 block(threads_per_block, 1, 1);

    void* args[] = {
        &d_input, 
        &this->d_output, 
        &this->dY_position, 
        &this->batch_size, &depth, &W, &W_out, &H, &H_out, &total_size
    };

    CUresult res = cuLaunchKernel(
        k_max_pooling_forward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0, 
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Max_Pooling] Kernel Launch Failed! Error code: " << res << std::endl;
    }

    // cuCtxSynchronize();

    return d_output;

}

float *Max_Pooling::backward(float *d_cache)
{
    if(d_cache == nullptr)
    {
        throw std::runtime_error("[Max_Pooling] Max_Pooling's cache requires the precede module's output memory");
    }
    if(d_cache_con == nullptr)
    {
        cudaMalloc(&d_cache_con, (size_t)in_size * sizeof(float));
    }
    
    int BLOCK_SIZE = 16;
    int grid_w = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_h = (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_w * grid_h, depth, this->batch_size);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

    void* args[] = {
        &d_cache, 
        &this->dY_position, 
        &this->d_cache_con, 
        &this->batch_size, &depth, &W_out, &W, &H_out, &H
    };

    cudaMemset(d_cache_con, 0, (this->in_size)*sizeof(float));

    CUresult res = cuLaunchKernel(
        k_max_pooling_backward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0, 
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Max_Pooling] Kernel Launch Failed! Error code: " << res << std::endl;
    }

    // cuCtxSynchronize();

    return d_cache_con;

}

