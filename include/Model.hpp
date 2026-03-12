#ifndef MODEL_HPP
#define MODEL_HPP

#include "Header.h"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Dataset.hpp"
#include <vector>

class Model
{
    private:
        std::vector<Layer*> pipeline;
        std::vector<CUstream> streams;
        // SGD_Momentum optimizer;
        Loss* loss_layer = nullptr;

        std::string dataset_path;
        int batch_size;
        
        float* d_input_buffer = nullptr;  
        size_t buffer_size = 0;
        int num_classes;

        Optimizer* optimizer = nullptr;
        OptimizerType opt_type;
    public:
        // Model(const std::string& dataset_path, int batch_size, int num_classes = 10, float learning_rate = 0.01f);
        // Model(const std::string& dataset_path, int batch_size, int num_classes = 10, float learning_rate = 0.01f, float momentum = 0.9f, float weight_decay = 0.0001f);
        Model(const std::string& dataset_path, int batch_size, int num_classes, float learning_rate, OptimizerType opt_type, float momentum = 0.9f, float weight_decay = 0.0001f);
        ~Model();


        void add_layer(Layer* in_layer);
        void set_loss(Loss* in_loss);
        void train(int epochs);
        void test(const std::string& test_path);
        void save_model();
};

#endif