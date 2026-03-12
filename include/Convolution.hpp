#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "Header.h"
#include "Layer.hpp"
#include "Optimizer.hpp"


class Convolution : public Layer
{
    public:
        float *weight = nullptr;
        float *bias = nullptr;

        float* d_weight = nullptr;
        float* d_bias = nullptr;

        int weight_size;
        int bias_size;

    
        Convolution(int in_channels, int out_channels, int image_height, int image_width, const std::string &filepath);
        ~Convolution();

        // float* forward(xt::xarray<float>& input, int batch_size);
        float* forward(float* d_input, int batch_size);
        float* backward(float* d_cache);
        void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) override;

        void init_weights(const std::string& filepath);
        void load_weight_bias();
        void save_weight_bias();

    private:

        void upload_params_to_gpu(const std::vector<float>& h_weights, const std::vector<float>& h_bias);
        
        CUfunction k_conv_forward;
        CUfunction k_conv_backward_weight;
        CUfunction k_conv_backward_bias;
        CUfunction k_conv_backward_input;

        int C_in, C_out, H_in, W_in;
        int H_out, W_out;
        int size;
        int batch_size;

        float* d_cache_con = nullptr;
        float *d_output = nullptr;
        float* d_input = nullptr;

        std::string filepath;
};
#endif