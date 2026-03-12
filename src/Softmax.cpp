#include "Softmax.hpp"
#include "Drive_Singleton.hpp"

float *Softmax::forward(float *d_input, int batch_size) 
{
    this->batch_size = batch_size;
    this->size = depth * 1 * 1 * batch_size;
    if(d_input == nullptr)
    {
        throw std::runtime_error("[Softmax] Softmax's input requires the precede module's output memory");
    }
    
    int BLOCK_SIZE = 32;
    dim3 grid(this->batch_size, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    void* args[] = {
        &d_input, 
        &this->batch_size,
        &this->depth,
        &this->size
    };
    
    CUresult res = cuLaunchKernel(
        k_softmax_forward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0, 
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Softmax] Kernel Launch Failed! Error code: " << res << std::endl;
    }

    // cuCtxSynchronize();

    return d_input;
}

float *Softmax::backward(float *d_cache)
{
    return d_cache;
}

Softmax::Softmax(int batch_size, int depth, int image_height = 1, int image_width = 1) : Layer()
{
    this->batch_size = batch_size;
    this->depth = depth;
    this->size = depth * image_height * image_width * batch_size;

    CUmodule mo = Driver_Singleton::getInstance()->getModule();
    CUresult res = cuModuleGetFunction(&k_softmax_forward, mo, "softmax_forward_kernel");

    if(res != CUDA_SUCCESS)
    {
        throw std::runtime_error("[Softmax] Cannot find kernel 'softmax_forward_kernel'");
    }
}

Softmax::~Softmax()
{
    // if(d_ouput == nullptr)
    // {
    //     cudaFree(d_ouput);
    //     d_ouput = nullptr;
    // }
}
