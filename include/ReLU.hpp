#ifndef RELU_HPP
#define RELU_HPP
#include "Header.h"
#include "Layer.hpp"
#include "Optimizer.hpp"

//For GPU so far...
class ReLU : public Layer
{
    public:
        float* forward(float* d_input, int batch_size);
        float* backward(float* d_input_cache);
        void save_weight_bias() override {}
        void load_weight_bias() override {}
        void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) override {}
        float* d_out;
        int batch_size;
        int depth;
        int H;
        int W;
        int total_size;
        //batch_size, out_channels, image_height, image_width
        ReLU(int batch_size, int depth, int image_height, int image_width);
        ~ReLU();

    private:
        CUfunction k_relu_forward;
        CUfunction k_relu_backward;
};
#endif