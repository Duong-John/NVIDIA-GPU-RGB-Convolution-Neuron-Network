#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP
#include "Header.h"
#include "Layer.hpp"
#include "Optimizer.hpp"

class Softmax : public Layer
{
    public:
        int batch_size;
        int size;
        int depth;
        // float* d_ouput = nullptr;
        float* forward(float* d_input, int batch_size);
        float* backward(float* d_cache);
        void save_weight_bias() override {}
        void load_weight_bias() override {}
        void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) override {}

        Softmax(int batch_size, int depth, int image_height, int image_width);
        ~Softmax();

    private:
        CUfunction k_softmax_forward;
};

#endif