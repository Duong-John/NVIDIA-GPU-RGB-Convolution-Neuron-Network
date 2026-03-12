#ifndef LOSS_HPP
#define LOSS_HPP
#include "Header.h"


class Loss
{
    public:

        float* d_input = nullptr; //No maclloc
        int* d_label = nullptr;
        float* d_cache = nullptr;
        float* loss_entropy = nullptr;
        int batch_size;
        int depth;
        float* forward(float* d_input, xt::xarray<int>& d_label);

        float* backward();
        float get_loss();
        Loss(int batch_size, int depth, int image_height = 1, int image_width = 1);
        ~Loss();
    
    private:
        CUfunction k_loss;
        CUfunction k_loss_backward;

};
#endif