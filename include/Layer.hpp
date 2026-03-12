#ifndef LAYER_HPP
#define LAYER_HPP


#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "Optimizer.hpp"

class Layer {
public:
    OptimizerType opt_type = OptimizerType::ADAM;
    bool is_training = true;
    virtual ~Layer() = default;
    virtual float* forward(float* d_input, int batch_size) = 0;
    virtual float* backward(float* d_cache) = 0;
    virtual void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) = 0;
    virtual void save_weight_bias() {}
    virtual void load_weight_bias() {}
    virtual void set_optimizer_type(OptimizerType type) 
    {
        this->opt_type = type;
    }
};
#endif
