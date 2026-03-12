#include "Linear.hpp"
#include "Drive_Singleton.hpp"

Linear::Linear(int batch_size, int flat_size, int neuron_num, const std::string &file_name) : Layer()
{
    this->batch_size = batch_size;
    this->flat_size = flat_size;
    this->neuron_num = neuron_num;
    this->file_name = file_name;
    this->weight_size = flat_size * neuron_num;
    this->bias_size = neuron_num;

    CUmodule mo = Driver_Singleton::getInstance()->getModule();
    CUresult res1 = cuModuleGetFunction(&k_linear_forward, mo, "linear_forward_kernel");
    CUresult res2 = cuModuleGetFunction(&k_linear_bias_backward, mo, "linear_bias_backward_kernel");
    CUresult res3 = cuModuleGetFunction(&k_linear_weight_backward, mo, "linear_weight_backward_kernel");
    CUresult res4 = cuModuleGetFunction(&k_linear_input_backward, mo, "linear_input_backward_kernel");

    if(res1 != CUDA_SUCCESS)
    {
        throw std::runtime_error("Cannot find kernel 'linear_forward_kernel'");
    }

    if(res2 != CUDA_SUCCESS)
    {
        throw std::runtime_error("[Linear] Cannot find kernel 'linear_bias_backward_kernel'");
    }

    if(res3 != CUDA_SUCCESS)
    {
        throw std::runtime_error("[Linear] Cannot find kernel 'linear_weight_backward_kernel'");
    }

    if(res4 != CUDA_SUCCESS)
    {
        throw std::runtime_error("[Linear] Cannot find kernel 'linear_input_backward_kernel'");
    }

    // load_weight_bias();

    cudaMalloc(&d_output, (size_t)batch_size * neuron_num * sizeof(float));
    cudaMalloc(&d_bias, (size_t) neuron_num * sizeof(float));
    cudaMalloc(&d_weight, (size_t)flat_size* neuron_num * sizeof(float));
    cudaMalloc(&d_cache_con, (size_t) batch_size * flat_size * sizeof(float));
}

Linear::~Linear()
{
    if (weight) 
    {
        cudaFree(weight);
        weight = nullptr;
    }
    if (bias) 
    {
        cudaFree(bias);
        bias = nullptr;
    }
    if (d_output) 
    {
        cudaFree(d_output);
        d_output = nullptr;
    }

    if (d_weight) 
    {
        cudaFree(d_weight);
        d_weight = nullptr;
    }
    if (d_bias) 
    {
        cudaFree(d_bias);
        d_bias = nullptr;
    }
    if (d_cache_con) 
    {
        cudaFree(d_cache_con);
        d_cache_con = nullptr;
    }
}

void Linear::update_params(Optimizer &optimizer, CUstream stream_W, CUstream stream_B)
{
    optimizer.learn(this->weight, this->d_weight, this->weight_size, stream_W);
    optimizer.learn(this->bias, this->d_bias, this->bias_size, stream_B);
}

float *Linear::forward(float *d_input, int batch_size)
{
    this->batch_size = batch_size;
    if (d_input == nullptr) {
        throw std::runtime_error("[Linear] Linear's input requires the precede module's output memory");
    }
    this->d_input = d_input;

    int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid(
        (neuron_num + block.x - 1) / block.x,
        (this->batch_size + block.y - 1) / block.y
    );

    void* args[] = {
        &d_input,
        &weight,
        &bias,
        &d_output,
        &this->batch_size,
        &flat_size,
        &neuron_num
    };

    CUresult res = cuLaunchKernel(
        k_linear_forward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0,
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Linear] Linear Kernel Launch Failed: " << res << std::endl;
    }

    // cuCtxSynchronize();

    return d_output;
}

float *Linear::backward(float *d_cache)
{
    if (d_cache == nullptr) {
        throw std::runtime_error("[Linear] Linear's input requires the precede module's output memory");
    }
    if (d_input == nullptr) {
        throw std::runtime_error("[Linear] Linear's input requires the precede module's output memory");
    }

    // 1. Calculate Bias:
    int NUM_BATCH_ROUND = 64;
    dim3 block_bias(NUM_BATCH_ROUND, 1, 1); 
    dim3 grid_bias(this->neuron_num, 1, 1);

    void* args1[] = {
        &d_cache,
        &d_bias,
        &batch_size,
        &neuron_num
    };

    CUresult res1 = cuLaunchKernel(
        k_linear_bias_backward,
        grid_bias.x, grid_bias.y, grid_bias.z,
        block_bias.x, block_bias.y, block_bias.z,
        0, 0,
        args1, 0
    );

    if (res1 != CUDA_SUCCESS) {
        std::cerr << "[Linear] Linear Kernel Launch Failed: " << res1 << std::endl;
    }

    //2. Calculate D_Weight:
    int BLOCK_SIZE = 16;
    int grid_W_x = (neuron_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_W_y = (flat_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid_W(grid_W_x, grid_W_y, 1);
    dim3 block_W(BLOCK_SIZE, BLOCK_SIZE, 1);

    void* args2[] = {
        &d_input,
        &d_cache,
        &d_weight,
        &batch_size,
        &flat_size,
        &neuron_num
    };

    CUresult res2 = cuLaunchKernel(
        k_linear_weight_backward,
        grid_W.x, grid_W.y, grid_W.z,
        block_W.x, block_W.y, block_W.z,
        0, 0,
        args2, 0
    );

    if (res2 != CUDA_SUCCESS) {
        std::cerr << "[Linear] Linear Kernel Launch Failed: " << res2 << std::endl;
    }

    //3. Calculate D_X:


    int grid_X_x = (flat_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_X_y = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_X(grid_X_x, grid_X_y, 1);
        
    dim3 block_X(BLOCK_SIZE, BLOCK_SIZE, 1);

    void* args3[] = {
        &d_cache,
        &weight,
        &d_cache_con,
        &batch_size,
        &flat_size,
        &neuron_num
    };

    CUresult res3 = cuLaunchKernel(
        k_linear_input_backward,
        grid_X.x, grid_X.y, grid_X.z,
        block_X.x, block_X.y, block_X.z,
        0, 0,
        args3, 0
    );

    if (res3 != CUDA_SUCCESS) {
        std::cerr << "[Linear] Linear Kernel Launch Failed: " << res3 << std::endl;
    }
    // cuCtxSynchronize();
    return this->d_cache_con;

}

void Linear::load_weight_bias()
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("[Linear] Cannot open weight file: " + file_name);
    }

    size_t weight_count = (size_t)flat_size * neuron_num;
    size_t bias_count = (size_t)neuron_num;

    std::vector<float> host_weights(weight_count);

    file.read(reinterpret_cast<char*>(host_weights.data()), weight_count * sizeof(float));
    cudaMalloc(&weight, weight_count * sizeof(float));
    cudaMemcpy(weight, host_weights.data(), weight_count * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> host_biases(bias_count);


    file.read(reinterpret_cast<char*>(host_biases.data()), bias_count * sizeof(float));
    //std::cout<<(int)weight_count;
    cudaMalloc(&bias, bias_count * sizeof(float));
    cudaMemcpy(bias, host_biases.data(), bias_count * sizeof(float), cudaMemcpyHostToDevice);
}

void Linear::save_weight_bias()
{
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_biases(bias_size);

    cudaMemcpy(h_weights.data(), weight, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_biases.data(), bias, bias_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(file_name, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("[Linear] Cannot open file to save: " + file_name);
    }

    file.write(reinterpret_cast<const char*>(h_weights.data()), weight_size * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_biases.data()), bias_size * sizeof(float));
    
    file.close();
    std::cout << "[Linear] Saved updated weights and biases to " << file_name << std::endl;
}
