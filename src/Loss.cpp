#include "Loss.hpp"
#include "Drive_Singleton.hpp"

float *Loss::forward(float *d_input, xt::xarray<int>& in_label)
{
    int current_batch_size = (int)in_label.size();
    this->batch_size = current_batch_size;
    if(this->d_label == nullptr)
    {
        size_t label_size = current_batch_size * sizeof(int);
        cudaMalloc(&this->d_label, label_size);
    }

    size_t copy_size = current_batch_size * sizeof(int);
    cudaMemcpy(this->d_label, in_label.data(), copy_size, cudaMemcpyHostToDevice);

    if (d_input == nullptr) {
        throw std::runtime_error("[Loss] Loss's input requires the precede module's output memory");
    }

    this->d_input = d_input;

    int BLOCK_SIZE = 64;
    dim3 grid(1, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    void* args[] = {
        &d_input,
        &d_label,
        &loss_entropy,
        &current_batch_size,
        &depth
    };
    // std::cout<<"Loss: ";
    CUresult res = cuLaunchKernel(
        k_loss,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0,
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Loss] Loss Kernel Launch Failed: " << res << std::endl;
    }

    cuCtxSynchronize();
    // std::cout<<std::endl;
    return loss_entropy;
}


float *Loss::backward()
{
    if(d_cache == nullptr)
    {
        size_t cache_size = this->batch_size * this->depth * sizeof(float);
        cudaMalloc(&this->d_cache, cache_size);
    }

    if(d_label == nullptr)
    {
        throw std::runtime_error("[Loss] Loss requires propagation before backpropagation: No Label");
    }

    if(d_input == nullptr)
    {
        throw std::runtime_error("[Loss] Loss requires propagation before backpropagation: No Input");
    }

    int BLOCK_SIZE = 32;
    dim3 grid(batch_size, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    void* args[] = {
        &d_input,
        &d_label,
        &d_cache,
        &this->batch_size,
        &depth
    };

    

    CUresult res = cuLaunchKernel(
        k_loss_backward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0,
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Loss] Loss Kernel Launch Failed: " << res << std::endl;
    }

    cuCtxSynchronize();
    
    return d_cache;


}

float Loss::get_loss()
{
    float h_loss = 0.0f;

    if (loss_entropy != nullptr)
    {
        cudaMemcpy(&h_loss, loss_entropy, sizeof(float), cudaMemcpyDeviceToHost);
    }
    else
    {
        throw std::runtime_error("[Loss] Loss output is null. Did you call forward() first?");
    }

    return h_loss;
}

Loss::Loss(int batch_size, int depth, int image_height, int image_width)
{
    this->batch_size = batch_size;
    this->depth = depth;

    CUmodule mo = Driver_Singleton::getInstance()->getModule();
    CUresult res1 = cuModuleGetFunction(&k_loss, mo, "loss_kernel");
    CUresult res2 = cuModuleGetFunction(&k_loss_backward, mo, "softmax_kernel_backward");

    if(res1 != CUDA_SUCCESS)
    {
        throw std::runtime_error("[Loss] Cannot find kernel 'loss_kernel'");
    }

    if(res2 != CUDA_SUCCESS)
    {
        throw std::runtime_error("[Loss] Cannot find kernel 'loss_kernel_backward'");
    }

    cudaMalloc(&this->loss_entropy, sizeof(float));
}

// Loss::Loss(int batch_size, int depth, int image_height = 1, int image_width = 1)
// {
//     this->batch_size = batch_size;
//     this->depth = depth;

//     CUmodule mo = Driver_Singleton::getInstance()->getModule();
//     CUresult res = cuModuleGetFunction(&k_loss, mo, "loss_kernel");

//     if(res != CUDA_SUCCESS)
//     {
//         throw std::runtime_error("Cannot find kernel 'loss_kernel'");
//     }

//     cudaMalloc(&this->loss_entropy, sizeof(float));
// }

Loss::~Loss()
{
    if(this->loss_entropy != nullptr)
    {
        cudaFree(loss_entropy);
        loss_entropy = nullptr;
    }
    if(this->d_label != nullptr)
    {
        cudaFree(d_label);
        d_label = nullptr;
    }
    if(this->d_cache != nullptr)
    {
        cudaFree(d_cache);
        d_cache = nullptr;
    }
}
