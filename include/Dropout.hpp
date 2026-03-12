#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "Header.h"
#include "Layer.hpp"


class Dropout : public Layer
{
    public:
        Dropout(int max_batch_size, int flat_size, float drop_rate = 0.5f);
        ~Dropout();

        float* forward(float* d_input, int batch_size);
        float* backward(float* d_cache);
        
        void save_weight_bias() override {}

        void load_weight_bias() override {}
        
        void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) override {}

    private:
        int max_size;
        int current_size;
        float drop_rate;
        
        float* d_mask = nullptr;
        void* d_states = nullptr;

        CUfunction k_dropout_init;
        CUfunction k_dropout_forward;
        CUfunction k_dropout_backward;
};
#endif