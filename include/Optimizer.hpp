#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Header.h"
#include <unordered_map>

enum class OptimizerType {
    SGD,
    SGD_MOMENTUM,
    ADAM
};

class Optimizer
{
    public:
        virtual ~Optimizer() = default;
        virtual void learn(float* param, float* grad, int size, CUstream hStream) = 0;
        virtual void set_learning_rate(float lr) = 0;
        virtual float get_learning_rate() = 0;
};

class SGD : public Optimizer
{
    private:
        float learning_rate;
        CUfunction k_sgd_update;

    public:
        SGD(float lr = 0.01f);
        ~SGD() override;
        void learn(float* param, float* grad, int size, CUstream hStream) override;
        void set_learning_rate(float lr) override;
        float get_learning_rate() override;
};

class SGD_Momentum : public Optimizer
{
    private:
        float learning_rate;
        float momentum;
        float weight_decay;
        CUfunction k_sgd_update;
        std::unordered_map<float*, float*> velocities;

    public:
        SGD_Momentum(float lr = 0.01f, float momentum = 0.9f, float weight_decay = 0.0001f); 
        ~SGD_Momentum() override;
        void learn(float* param, float* grad, int size, CUstream hStream) override;
        void set_learning_rate(float lr) override;
        float get_learning_rate() override;
};

class Adam : public Optimizer
{
    private:
        float learning_rate;
        float beta1;
        float beta2;
        float epsilon;
        float weight_decay;
        CUfunction k_adam_update;

        std::unordered_map<float*, float*> m_velocities;
        std::unordered_map<float*, float*> v_velocities;
        std::unordered_map<float*, int> t_steps;

    public:
        Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.0001f);
        ~Adam() override;
        void learn(float* param, float* grad, int size, CUstream hStream) override;
        void set_learning_rate(float lr) override;
        float get_learning_rate() override;
};

#endif