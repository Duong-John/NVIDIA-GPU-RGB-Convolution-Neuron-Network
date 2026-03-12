#include "Optimizer.hpp"
#include "Drive_Singleton.hpp"

SGD::SGD(float lr) : learning_rate(lr)
{
    CUmodule mod = Driver_Singleton::getInstance()->getModule();
    CUresult res = cuModuleGetFunction(&k_sgd_update, mod, "sgd_update_kernel");

    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("[Optimizer] Cannot find kernel 'sgd_update_kernel' in PTX");
    }
}

SGD::~SGD()
{
}

void SGD::learn(float* param, float* grad, int size, CUstream hStream)
{
    if (param == nullptr || grad == nullptr || size == 0) return;

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    void* args[] = {
        &param,
        &grad,
        &this->learning_rate,
        &size
    };

    cuLaunchKernel(
        k_sgd_update,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, hStream, args, 0
    );
}

void SGD_Momentum::set_learning_rate(float lr) {
    this->learning_rate = lr;
}

float SGD_Momentum::get_learning_rate() {
    return this->learning_rate;
}

// void SGD::learn(float* weight, float* bias, float* d_weight, float* d_bias, int weight_size, int bias_size)
// {
//     update_array(weight, d_weight, weight_size);
//     update_array(bias, d_bias, bias_size);
    
//     cuCtxSynchronize(); 
// }

SGD_Momentum::SGD_Momentum(float lr, float momentum, float weight_decay): learning_rate(lr), momentum(momentum), weight_decay(weight_decay)
{
    CUmodule mod = Driver_Singleton::getInstance()->getModule();

    CUresult res = cuModuleGetFunction(&k_sgd_update, mod, "sgd_momentum_update_kernel");

    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot find kernel 'sgd_momentum_update_kernel' in PTX");
    }
}

SGD_Momentum::~SGD_Momentum()
{
    
    for (auto const& [param_ptr, velocity_ptr] : velocities) 
    {
        if (velocity_ptr != nullptr) {
            cudaFree(velocity_ptr);
        }
    }
    std::cout << "[Optimizer] Released all velocity memory." << std::endl;
}

void SGD_Momentum::learn(float *param, float *grad, int size, CUstream hStream)
{
    if (param == nullptr || grad == nullptr || size == 0) return;

    if (velocities.find(param) == velocities.end()) 
    {
        float* d_vel;
        cudaMalloc(&d_vel, size * sizeof(float));
        
        cudaMemsetAsync(d_vel, 0, size * sizeof(float), hStream);
        
        velocities[param] = d_vel;
    }

    float* d_vel = velocities[param];

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    void* args[] = {
        &param,
        &grad,
        &d_vel,
        &this->learning_rate,
        &this->momentum,
        &this->weight_decay,
        &size
    };

    cuLaunchKernel(
        k_sgd_update,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 
        hStream, 
        args, 0
    );
}

void SGD::set_learning_rate(float lr) {
    this->learning_rate = lr;
}

float SGD::get_learning_rate() {
    return this->learning_rate;
}

Adam::Adam(float lr, float beta1, float beta2, float epsilon, float weight_decay)
    : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay)
{
    CUmodule mod = Driver_Singleton::getInstance()->getModule();
    CUresult res = cuModuleGetFunction(&k_adam_update, mod, "adam_update_kernel");

    if (res != CUDA_SUCCESS) 
    {
        throw std::runtime_error("Cannot find kernel 'adam_update_kernel' in PTX");
    }
}

Adam::~Adam()
{
    for (auto const& [param_ptr, m_ptr] : m_velocities) 
    {
        if (m_ptr != nullptr) cudaFree(m_ptr);
    }
    for (auto const& [param_ptr, v_ptr] : v_velocities) 
    {
        if (v_ptr != nullptr) cudaFree(v_ptr);
    }
    std::cout << "[Optimizer] Released all Adam memory." << std::endl;
}

void Adam::learn(float *param, float *grad, int size, CUstream hStream)
{
    if (param == nullptr || grad == nullptr || size == 0) return;

    if (m_velocities.find(param) == m_velocities.end()) 
    {
        float* d_m; float* d_v;
        cudaMalloc(&d_m, size * sizeof(float));
        cudaMalloc(&d_v, size * sizeof(float));
        
        cudaMemsetAsync(d_m, 0, size * sizeof(float), hStream);
        cudaMemsetAsync(d_v, 0, size * sizeof(float), hStream);
        
        m_velocities[param] = d_m;
        v_velocities[param] = d_v;
        t_steps[param] = 0;
    }

    float* d_m = m_velocities[param];
    float* d_v = v_velocities[param];
    
    t_steps[param]++;
    int t = t_steps[param];

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    void* args[] = {
        &param, &grad, &d_m, &d_v,
        &this->learning_rate, &this->beta1, &this->beta2, &this->epsilon, &this->weight_decay,
        &t, &size
    };

    cuLaunchKernel(
        k_adam_update,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, hStream, args, 0
    );
}

void Adam::set_learning_rate(float lr) {
    this->learning_rate = lr;
}

float Adam::get_learning_rate() {
    return this->learning_rate;
}