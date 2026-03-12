#ifndef LINEAR_HPP
#define LINEAR_HPP
#include "Header.h"
#include "Layer.hpp"
#include "Optimizer.hpp"

class Linear : public Layer
{
    private:


        float* d_output = nullptr;
        float* d_cache_con = nullptr;
        float* d_input = nullptr;

        int batch_size;
        int flat_size;
        int neuron_num;

        std::string file_name;
        CUfunction k_linear_forward;
        CUfunction k_linear_bias_backward;
        CUfunction k_linear_weight_backward;
        CUfunction k_linear_input_backward;
        
        
    
    public:

        Linear(int batch_size, int flat_size, int neuron_num, const std::string& file_name);
        ~Linear();
        void update_params(Optimizer& optimizer, CUstream stream_W, CUstream stream_B) override;
        float* forward(float* d_input, int batch_size);
        float* backward(float* d_cache);

        void save_weight_bias();
        void load_weight_bias();
        

        float* weight = nullptr;
        float* bias = nullptr;

        float* d_bias = nullptr;
        float* d_weight = nullptr;

        int weight_size;
        int bias_size;

};
#endif