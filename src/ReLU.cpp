#include "ReLU.hpp"
#include "Drive_Singleton.hpp"

ReLU::ReLU(int batch_size, int depth, int image_height, int image_width) : Layer()
{
    this->batch_size = batch_size;
    this->depth = depth;
    this->H = image_height;
    this->W = image_width;
    this->total_size = batch_size * depth * image_height * image_width;
    CUmodule mod = Driver_Singleton::getInstance()->getModule();
    CUresult res1 = cuModuleGetFunction(&k_relu_forward, mod, "relu_forward_kernel");
    CUresult res2 = cuModuleGetFunction(&k_relu_backward, mod, "relu_backward_kernel");

    if (res1 != CUDA_SUCCESS) {
        throw std::runtime_error("[ReLU] Cannot find kernel 'conv_forward_kernel' in PTX");
    }

    if (res2 != CUDA_SUCCESS) {
        throw std::runtime_error("[ReLU] Cannot find kernel 'conv_backward_kernel' in PTX");
    }
}

float *ReLU::forward(float *d_input, int batch_size)
{
    if(d_input == nullptr)
    {
        throw std::runtime_error("[ReLU] ReLU's input requires the precede module's output memory");
    }

    this->total_size = batch_size * depth * H * W;

    int threads_per_block = 256;

    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    void* args[] = {&d_input, &this->total_size};

    CUresult res = cuLaunchKernel(
        k_relu_forward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0, 
        args, 0
    );

    //Wait all threads to finish
    // cuCtxSynchronize();

    if (res != CUDA_SUCCESS) {
        std::cerr << "[ReLU] Kernel Launch Failed! Error code: " << res << std::endl;
    }
    this->d_out = d_input;
    return d_input;
}

float *ReLU::backward(float *d_input_cache)
{
    if(d_input_cache == nullptr)
    {
        throw std::runtime_error("[ReLU] ReLU's cache requires the precede module's output memory");
    }

    
    int threads_per_block = 256;

    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    void* args[] = {&d_out, &d_input_cache, &this->total_size};

    CUresult res = cuLaunchKernel(
        k_relu_backward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0, 
        args, 0
    );

    //Wait all threads to finish
    // cuCtxSynchronize();

    if (res != CUDA_SUCCESS) {
        std::cerr << "[ReLU] Kernel Launch Failed! Error code: " << res << std::endl;
    }

    return d_input_cache;
}


ReLU::~ReLU()
{
}
