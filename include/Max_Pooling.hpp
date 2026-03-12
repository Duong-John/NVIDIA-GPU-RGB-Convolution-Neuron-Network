#ifndef MAX_POOLING_HPP
#define MAX_POOLING_HPP
#include "Header.h"
#include "Layer.hpp"
#include "Optimizer.hpp"

class Max_Pooling : public Layer
{
    public:
        int batch_size;
        int depth;
        int H;
        int W;
        int total_size;
        int in_size;
        int H_out;
        int W_out;
        
        Max_Pooling(int batch_size, int depth, int image_height, int image_width);
        ~Max_Pooling();

        float* forward(float* input, int batch_size);
        float* backward(float* d_cache);
        void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) override {}

        void save_weight_bias() override {}
        void load_weight_bias() override {}

    private:
        int* dY_position = nullptr;
        float* d_output = nullptr;
        float* d_cache_con = nullptr;
        
        CUfunction k_max_pooling_forward;
        CUfunction k_max_pooling_backward;

};
#endif