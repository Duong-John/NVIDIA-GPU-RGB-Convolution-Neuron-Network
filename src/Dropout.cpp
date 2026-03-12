#include "Dropout.hpp"
#include "Drive_Singleton.hpp"

Dropout::Dropout(int max_batch_size, int flat_size, float drop_rate) : Layer()
{
    this->drop_rate = drop_rate;
    this->max_size = max_batch_size * flat_size;

    CUmodule mod = Driver_Singleton::getInstance()->getModule();
    cuModuleGetFunction(&k_dropout_init, mod, "dropout_init_kernel");
    cuModuleGetFunction(&k_dropout_forward, mod, "dropout_forward_kernel");
    cuModuleGetFunction(&k_dropout_backward, mod, "dropout_backward_kernel");


    cudaMalloc(&d_mask, max_size * sizeof(float));

    size_t state_size = 48; 
    cudaMalloc(&this->d_states, max_size * state_size);


    int threads_per_block = 256;
    int num_blocks = (max_size + threads_per_block - 1) / threads_per_block;
    dim3 grid(num_blocks, 1, 1);

    dim3 block(threads_per_block, 1, 1);
    int seed = 1234; 

    void* args_init[] = { &seed, &d_states, &max_size };

    cuLaunchKernel(k_dropout_init, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, 0, args_init, 0);
    cuCtxSynchronize();
}

Dropout::~Dropout()
{
    if (d_mask) cudaFree(d_mask);
    if (d_states) cudaFree(d_states);
}

float *Dropout::forward(float *d_input, int batch_size)
{
    this->current_size = batch_size * (max_size / (this->max_size / batch_size));

    int threads_per_block = 256;

    int num_blocks = (current_size + threads_per_block - 1) / threads_per_block;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    void* args[] = {
        &d_input, 
        &d_mask, 
        &d_states, 
        &this->drop_rate, 
        &this->current_size, 
        &this->is_training
    };

    cuLaunchKernel(k_dropout_forward, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, 0, args, 0);
    return d_input;
}

float *Dropout::backward(float *d_cache)
{
    int threads_per_block = 256;
    int num_blocks = (current_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(num_blocks, 1, 1);

    dim3 block(threads_per_block, 1, 1);

    void* args[] = { &d_cache, &d_mask, &this->current_size };
    cuLaunchKernel(k_dropout_backward, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, 0, args, 0);
    
    return d_cache;
}