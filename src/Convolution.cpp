#include "Convolution.hpp"
#include "Drive_Singleton.hpp"

// extern "C" void launch_conv_kernel(
//     float* d_X, float* d_W, float* d_B, float* d_Y,
//     int N, int Cin, int H, int Width,
//     int Cout, int H_out, int W_out
// );

Convolution::Convolution(int in_channels, int out_channels, int image_height, int image_width, const std::string &filepath) : Layer()
{
    this->C_in = in_channels;
    this->C_out = out_channels;
    this->H_in = image_height;
    this->W_in = image_width;
    // this->weight_size = 3 * 3 * out_channels;
    // this->bias_size = out_channels;
    this->filepath = filepath;
    
    CUmodule mod = Driver_Singleton::getInstance()->getModule();
    CUresult res1 = cuModuleGetFunction(&k_conv_forward, mod, "conv_forward_kernel");
    CUresult res2 = cuModuleGetFunction(&k_conv_backward_weight, mod, "conv_weight_backward_naive");
    CUresult res3 = cuModuleGetFunction(&k_conv_backward_bias, mod, "conv_bias_backward_naive");
    CUresult res4 = cuModuleGetFunction(&k_conv_backward_input, mod, "conv_input_backward_naive");
    
    if (res1 != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot find kernel 'conv_forward_kernel' in PTX");
    }

    if (res2 != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot find kernel 'conv_weight_backward_naive' in PTX");
    }

    if (res3 != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot find kernel 'conv_bias_backward_naive' in PTX");
    }

    if (res4 != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot find kernel 'conv_input_backward_naive' in PTX");
    }
    
    H_out = H_in - 2;
    W_out = W_in - 2;

    size_t w_size = C_out*C_in*3*3*sizeof(float);
    size_t b_size = C_out*sizeof(float);

    // cudaMalloc(&d_weights, w_size);
    // cudaMalloc(&d_bias, b_size);

    cudaMalloc(&d_weight, w_size);
    cudaMalloc(&d_bias, b_size);

    // load_weight_bias();
}

Convolution::~Convolution()
{
    if(weight)
    {
        cudaFree(weight);
        weight = nullptr;
    }

    if(bias)
    {
        cudaFree(bias);
        bias = nullptr;
    }

    if(d_output)
    {
        cudaFree(d_output);
        d_output = nullptr;
    }

    if(d_input)
    {
        cudaFree(d_input);
        d_input = nullptr;
    }

    if(d_weight) 
    {
        cudaFree(d_weight);
    }
    if(d_bias) 
    {
        cudaFree(d_bias);
    }
    if(d_cache_con) 
    {
        cudaFree(d_cache_con);
    }
    
    
    std::cout << "[Convolution] GPU memory released successfully." << std::endl;
}

float *Convolution::forward(float* d_input, int batch_size)
{
    this->batch_size = batch_size;
    
    if (d_input == nullptr) {
        throw std::runtime_error("[Convolution] Convolution's input requires valid GPU memory");
    }
    this->d_input = d_input;

    if (d_output == nullptr) {
        size_t out_size = batch_size * C_out * H_out * W_out * sizeof(float);
        cudaMalloc(&d_output, out_size);
    }

    int N = batch_size;
    int _Cin = C_in; 
    int _H = H_in; 
    int _W = W_in;
    int _Cout = C_out;
    int _Ho = H_out; 
    int _Wo = W_out;

    void* args[] = {
        &this->d_input,
        &weight,
        &bias,
        &d_output,
        &N, &_Cin, &_H, &_W,
        &_Cout, &_Ho, &_Wo
    };

    int BLOCK_SIZE = 16;
    int grid_w = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_h = (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_w * grid_h, C_out, N);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

    CUresult res = cuLaunchKernel(
        k_conv_forward,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, 0, 
        args, 0
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[Convolution] Kernel Launch Failed! Error code: " << res << std::endl;
    }

    // cuCtxSynchronize();
    
    return d_output;
}

// float *Convolution::forward(xt::xarray<float>& input, int batch_size)
// {
//     this->batch_size = batch_size;
//     size_t input_size = input.size() * sizeof(float);

//     if (d_input != nullptr) {
//         cudaFree(d_input);
//         d_input = nullptr;
//     }
    
//     cudaMalloc(&d_input, input_size);
//     cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);

//     if (d_output == nullptr) {
//         size_t out_size = batch_size * C_out * H_out * W_out * sizeof(float);
//         cudaMalloc(&d_output, out_size);
//     }

//     int N = batch_size;
//     int _Cin = C_in; 
//     int _H = H_in; 
//     int _W = W_in;
//     int _Cout = C_out;
//     int _Ho = H_out; 
//     int _Wo = W_out;

//     void* args[] = {
//         &d_input,
//         &weight,
//         &bias,
//         &d_output,
//         &N, &_Cin, &_H, &_W,
//         &_Cout, &_Ho, &_Wo
//     };

//     int BLOCK_SIZE = 16;
//     int grid_w = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     int grid_h = (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE;

//     dim3 grid(grid_w * grid_h, C_out, N);
//     dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

//     // dim3 block(32, 1, 1);
//     // dim3 grid(H_out, C_out, batch_size);

//     CUresult res = cuLaunchKernel(
//         k_conv_forward,
//         grid.x, grid.y, grid.z,
//         block.x, block.y, block.z,
//         0, 0, 
//         args, 0
//     );

//     if (res != CUDA_SUCCESS) {
//         std::cerr << "Kernel Launch Failed! Error code: " << res << std::endl;
//     }

//     cuCtxSynchronize();

//     return d_output;


// }

float *Convolution::backward(float *d_cache)
{
    this->size = C_in * batch_size * H_in * W_in;
    if(d_cache == nullptr)
    {
        throw std::runtime_error("[Convolution] Convolution's cache requires the precede module's output memory");
    }

    if(d_cache_con == nullptr)
    {
        cudaMalloc(&d_cache_con, (size_t)size * sizeof(float));
    }

    //1. dW
    dim3 block_W(3, 3, 1);
    dim3 grid_W(C_out, C_in, 1);

    void* args_W[] = {
        &d_input,
        &d_cache,     
        &d_weight,
        &batch_size, &C_in, &H_in, &W_in,
        &C_out, &H_out, &W_out
    };

    CUresult res1 = cuLaunchKernel(
        k_conv_backward_weight,
        grid_W.x, grid_W.y, grid_W.z,
        block_W.x, block_W.y, block_W.z,
        0, 0, args_W, 0
    );

    if (res1 != CUDA_SUCCESS) {
        std::cerr << "[Convolution] Kernel Launch Failed! Error code: " << res1 << std::endl;
    }

    // 2. dB
    int threads_per_block = 256;
    int blocks_per_grid = (C_out + threads_per_block - 1) / threads_per_block;

    dim3 block_B(threads_per_block, 1, 1);
    dim3 grid_B(blocks_per_grid, 1, 1);

    void* args_B[] = {
        &d_cache,   
        &d_bias,
        &batch_size,
        &C_out, &H_out, &W_out
    };

    CUresult res2 = cuLaunchKernel(
        k_conv_backward_bias,
        grid_B.x, grid_B.y, grid_B.z,
        block_B.x, block_B.y, block_B.z,
        0, 0, args_B, 0
    );

    if (res2 != CUDA_SUCCESS) {
        std::cerr << "[Convolution] Kernel Launch Failed! Error code: " << res2 << std::endl;
    }

    
    // 3. dX

    int BLOCK_SIZE = 16;
    int grid_X_w = (W_in + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_X_h = (H_in + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid_X(grid_X_w * grid_X_h, C_in, batch_size);
    dim3 block_X(BLOCK_SIZE, BLOCK_SIZE, 1);

    void* args_X[] = {
        &d_cache,
        &weight,
        &d_cache_con,
        &batch_size, 
        &C_in, &H_in, &W_in,
        &C_out, &H_out, &W_out
    };

    CUresult res_X = cuLaunchKernel(
        k_conv_backward_input,
        grid_X.x, grid_X.y, grid_X.z,
        block_X.x, block_X.y, block_X.z,
        0, 0, args_X, 0
    );

    // cuCtxSynchronize();

    return this->d_cache_con;
    
}

void Convolution::update_params(Optimizer &optimizer, CUstream stream_W, CUstream stream_B)
{
    optimizer.learn(this->weight, this->d_weight, this->weight_size, stream_W);
    optimizer.learn(this->bias, this->d_bias, this->bias_size, stream_B);
}

void Convolution::init_weights(const std::string &filepath)
{
    size_t w_count = C_out * C_in * 3 * 3;
    size_t b_count = C_out;

    std::vector<float> h_weights(w_count);
    std::vector<float> h_bias(b_count);


    float stddev = std::sqrt(2.0f / (C_in * 3 * 3));
    std::random_device rd;


    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    for(size_t i = 0; i < w_count; ++i) h_weights[i] = dist(gen);

    std::fill(h_bias.begin(), h_bias.end(), 0.01f);


    std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "[Convolution] Error: Cannot write to file " << filepath << std::endl;
        return;
    }


    file.write(reinterpret_cast<const char*>(h_weights.data()), w_count * sizeof(float));

    file.write(reinterpret_cast<const char*>(h_bias.data()), b_count * sizeof(float));
    
    file.close();
    std::cout << "[Convolution] Initialized random weights and saved to " << filepath << std::endl;

    upload_params_to_gpu(h_weights, h_bias);
}

void Convolution::load_weight_bias()
{
    size_t w_count = C_out * C_in * 3 * 3;
    size_t b_count = C_out;

    this->weight_size = (int)w_count;
    this->bias_size = (int)C_out;

    std::vector<float> h_weights(w_count);
    std::vector<float> h_bias(b_count);

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open weight file " + filepath);
    }


    file.read(reinterpret_cast<char*>(h_weights.data()), w_count * sizeof(float));
    if (!file) throw std::runtime_error("Error: File size too small for weights");

    
    file.read(reinterpret_cast<char*>(h_bias.data()), b_count * sizeof(float));
    if (!file) throw std::runtime_error("Error: File size too small for bias");

    file.close();
    std::cout << "[Convolution] Loaded weights from " << filepath << std::endl;

    
    upload_params_to_gpu(h_weights, h_bias);
}

void Convolution::save_weight_bias()
{
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_biases(bias_size);

    cudaMemcpy(h_weights.data(), weight, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_biases.data(), bias, bias_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("[Convolution] Cannot open file to save: " + filepath);
    }

    file.write(reinterpret_cast<const char*>(h_weights.data()), weight_size * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_biases.data()), bias_size * sizeof(float));
    
    file.close();
    std::cout << "[Convolution] Saved updated weights and biases to " << filepath << std::endl;
}

void Convolution::upload_params_to_gpu(const std::vector<float> &h_weights, const std::vector<float> &h_bias)
{
    size_t w_size = h_weights.size() * sizeof(float);

    size_t b_size = h_bias.size() * sizeof(float);



    if(weight == nullptr) cudaMalloc(&weight, w_size);
    if(bias == nullptr) cudaMalloc(&bias, b_size);

    cudaMemcpy(weight, h_weights.data(), w_size, cudaMemcpyHostToDevice);

    cudaMemcpy(bias, h_bias.data(), b_size, cudaMemcpyHostToDevice);
}
