#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>
// #include <device_launch_parameters.h>
// #include <cstdio>

#define BLOCK_SIZE 16
#define INPUT_TILE_SIZE (BLOCK_SIZE + 2)

extern "C"
__global__ void conv_forward_kernel(
    const float* __restrict__ X,  // Input:  [N, Cin, H, W]
    const float* __restrict__ W,  // Weight: [Cout, Cin, 3, 3]
    const float* __restrict__ B,  // Bias:   [Cout]
    float* __restrict__ Y,        // Output: [N, Cout, H_out, W_out]
    int N, int C_in, int H, int Width, 
    int C_out, int H_out, int W_out
) 
{
    __shared__ float s_input[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int n = blockIdx.z; //Batch - which image ?
    int m = blockIdx.y; //Filter Output - kernel ?

    int grid_w = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int base_row = blockIdx.x / grid_w * BLOCK_SIZE;
    int base_col = blockIdx.x % grid_w * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = base_row + ty;
    int col_o = base_col + tx;

    float acc = 0.0f;

    for(int c = 0; c < C_in; ++c)
    {

        // int input_row_start = (blockIdx.x / ((W_out + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) - 1;
        // int input_col_start = (blockIdx.x % ((W_out + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) - 1;
        
        int tid = ty * BLOCK_SIZE + tx; // ID thread (0-255) (ty, tx 0-15)

        //Grid-Stride Loop: Extra load
        for (int i = tid; i < INPUT_TILE_SIZE * INPUT_TILE_SIZE; i += BLOCK_SIZE * BLOCK_SIZE)
        {
            //s_y, s_x: where to place in local shared mem
            int s_y = i / INPUT_TILE_SIZE; 
            int s_x = i % INPUT_TILE_SIZE;

            //base_row, base_col: acttual pixel in image to be loaded into shared mem
            //g_row, g_col: -1 because expanded shared mem, +s_y +s_x to shift up, down, left, right and may read
            //the same pixel that other blocks read because pixel is owned by the image, which is global

            //More description in related document: Tilling technique
            int g_row = base_row - 1 + s_y; 
            int g_col = base_col - 1 + s_x;

            if (g_row >= 0 && g_row < H && g_col >= 0 && g_col < Width) 
            {
                int input_idx = n * (C_in * H * Width) + c * (H * Width) + g_row * Width + g_col;
                s_input[s_y][s_x] = X[input_idx];
            } 
            else 
            {
                s_input[s_y][s_x] = 0.0f;
            }
        }  

        __syncthreads();

        if(row_o < H_out && col_o < W_out)
        {
            const float* w_ptr = W + (m*C_in*9) + (c*9);

            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    //In shared memory
                    float val = s_input[threadIdx.y + i][threadIdx.x + j];

                    acc += val * w_ptr[i * 3 + j];
                }
            }
        }
        __syncthreads();
    }
    if (row_o < H_out && col_o < W_out) 
    {
        Y[n * (C_out * H_out * W_out) + m * (H_out * W_out) + row_o * W_out + col_o] = acc + B[m];
    }
}

extern "C"
__global__ void conv_forward_naive(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int N, int Cin, int H_in, int W_in, 
    int Cout, int H_out, int W_out
) 
{
    int n = blockIdx.z; // Batch item
    int m = blockIdx.y; // Output Channel (Filter)
    int row_o = blockIdx.x; // Output Row
    int col_o = threadIdx.x; // Output Col


    if (row_o < H_out && col_o < W_out)
    {
        float acc = 0.0f;

        
        for(int c = 0; c < Cin; ++c)
        {
            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    int in_row = row_o + i;
                    int in_col = col_o + j;

                    int input_idx = n * (Cin * H_in * W_in) + c * (H_in * W_in) + in_row * W_in + in_col;
                    int weight_idx = m * (Cin * 9) + c * 9 + i * 3 + j;

                    acc += X[input_idx] * W[weight_idx];
                }
            }
        }

        int output_idx = n * (Cout * H_out * W_out) + m * (H_out * W_out) + row_o * W_out + col_o;
        Y[output_idx] = acc + B[m];
    }
}


extern "C" 
__global__ void relu_forward_kernel(float* X, int size)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_index < size)
    {
        float val = X[thread_index];
        // X[thread_index] = fmaxf(0.0f, val);
        X[thread_index] = (val > 0.0f) ? val : 0.01f * val;
        // if(val < 0.0f) X[thread_index] = 0.0f;
    }

}

//No memory usage
extern "C" 
__global__ void relu_backward_kernel(
    const float* X_out, 
    float* dX,
    int size)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_index < size)
    {
        float value = X_out[thread_index];
        if (value <= 0.0f)
        {
            // dX[thread_index] = 0.0f;
            dX[thread_index] *= 0.01f;
        }
    }

    // else
    // {
    //     printf("%f", value);
    // }

}

#define STRIDE 2
extern "C"
__global__ void max_pooling_forward_kernel_1D(
    const float* __restrict__ X, 
    float* Y, 
    int* dY_position, 
    int batch, int C, int W_in, int W_out, int H_in, int H_out, int size)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x; //thread_index of ouput
    if(thread_index < size)
    {

        int temp_idx = thread_index;
        
        int w_out = temp_idx % W_out;
        temp_idx /= W_out;
        
        int h_out = temp_idx % H_out;
        temp_idx /= H_out;
        
        int c = temp_idx % C;
        int n = temp_idx / C;

        int h_in_start = h_out * STRIDE;
        int w_in_start = w_out * STRIDE;

        int input_offset = n * (C * H_in * W_in) + c * (H_in * W_in);
        
        int input_idx = input_offset + h_in_start * W_in + w_in_start;

        float max = X[input_idx];
        int position = input_idx;

        float v1 = X[input_idx + 1];
        if(v1 > max)
        {
            max = v1;
            position = input_idx + 1;
        }

        float v2 = X[input_idx + W_in];
        if(v2 > max)
        {
            max = v2;
            position = input_idx + W_in;
        }

        float v3 = X[input_idx + W_in + 1];
        if(v3 > max)
        {
            max = v3;
            position = input_idx + W_in + 1;
        }

        Y[thread_index] = max;
        dY_position[thread_index] = position;  
    }
    
}

extern "C" 
__global__ void max_pooling_backward_kernel_2D(
    const float* __restrict__ dZ,
    const int* __restrict__ dY_position,
    float* dX,
    int batch, int C, int dZ_W, int dX_W, int dZ_H, int dX_H
)
{
    int n = blockIdx.z; //Batch - which image ?
    int c = blockIdx.y; //Filter Output - kernel ?

    int grid_w = (dZ_W + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int base_row = blockIdx.x / grid_w * BLOCK_SIZE;
    int base_col = blockIdx.x % grid_w * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_dZ = base_row + ty;
    int col_dZ = base_col + tx;

    if (row_dZ < dZ_H && col_dZ < dZ_W)
    {
        // int row_dX_start = row_dZ * STRIDE; 
        // int col_dX_start = col_dZ * STRIDE; 

        int dZ_offset = n * (C * dZ_H * dZ_W) + c * (dZ_H * dZ_W);
        // int dX_offset = n * (C * dX_H * dX_W) + c * (dX_H * dX_W);

        int input_idx = dZ_offset + row_dZ * dZ_W + col_dZ;
        // int output_idx = dX_offset + row_dX_start * dX_W + col_dX_start;

        // // dX[output_idx] = 0.0f;
        // // dX[output_idx + 1] = 0.0f;
        // // dX[output_idx + dX_W] = 0.0f;
        // // dX[output_idx + dX_W + 1] = 0.0f;

        int pos = dY_position[input_idx];
        dX[pos] = dZ[input_idx];

    }
}

extern "C"
__global__ void max_pooling_forward_kernel_2D(
    const float* __restrict__ X, 
    float* __restrict__ Y, 
    int* __restrict__ dY_position, 
    int batch, int C, int W_in, int W_out, int H_in, int H_out, int size)
{
    int n = blockIdx.z; //Batch - which image ?
    int c = blockIdx.y; //Filter Output - kernel ?

    int grid_w = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int base_row = blockIdx.x / grid_w * BLOCK_SIZE;
    int base_col = blockIdx.x % grid_w * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = base_row + ty;
    int col_o = base_col + tx;

    //Like the above convolution forwarding
    if (row_o < H_out && col_o < W_out)
    {
        int h_in_start = row_o * STRIDE;
        int w_in_start = col_o * STRIDE;

        int input_offset = n * (C * H_in * W_in) + c * (H_in * W_in);
        int output_offset = n * (C * H_out * W_out) + c * (H_out * W_out);
            
        int input_idx = input_offset + h_in_start * W_in + w_in_start;
        int output_idx = output_offset + row_o * W_out + col_o;

        float max = X[input_idx];
        int position = input_idx;

        // float v1 = X[input_idx + 1];
        // if(v1 > max)
        // {
        //     max = v1;
        //     position = input_idx + 1;
        // }

        // float v2 = X[input_idx + W_in];
        // if(v2 > max)
        // {
        //     max = v2;
        //     position = input_idx + W_in;
        // }

        // float v3 = X[input_idx + W_in + 1];
        // if(v3 > max)
        // {
        //     max = v3;
        //     position = input_idx + W_in + 1;
        // }

        if (w_in_start + 1 < W_in) 
        {
            float v1 = X[input_idx + 1];
            if(v1 > max) 
            { 
                max = v1; 
                position = input_idx + 1; 
            }
        }

        if (h_in_start + 1 < H_in) 
        {
            float v2 = X[input_idx + W_in];
            if(v2 > max) 
            { 
                max = v2; 
                position = input_idx + W_in; 
            }
        }

        if (w_in_start + 1 < W_in && h_in_start + 1 < H_in) 
        {
            float v3 = X[input_idx + W_in + 1];
            if(v3 > max) 
            { 
                max = v3; 
                position = input_idx + W_in + 1; 
            }
        }

        Y[output_idx] = max;
        dY_position[output_idx] = position;  
    }
}

extern "C"
__global__ void linear_forward_kernel(
    const float* __restrict__ X,    // Input [Batch, Flat_Size]
    const float* __restrict__ W,    // Weight [Flat_Size, Neuron_Num]
    const float* __restrict__ B,    // Bias [Neuron_Num]
    float* __restrict__ Y,          // Output [Batch, Neuron_Num]
    int batch_size,
    int flat_size,
    int neuron_num
)
{
    
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < neuron_num)
    {
        float acc = 0.0f;

        for (int k = 0; k < flat_size; ++k)
        {
            float x_val = X[row * flat_size + k];
            
            float w_val = W[k * neuron_num + col];
            //printf("%d", neuron_num);
            acc += x_val * w_val;
        }

        Y[row * neuron_num + col] = acc + B[col];
    }
}

#define NUM_NEURON_ROUND 32
#define NEG_INF -1e9f
extern "C" 
__global__ void softmax_forward_kernel(
    float* X,
    int batch_size,
    int neuron_num,
    int input_size
)
{
    __shared__ float s_data[NUM_NEURON_ROUND][2];
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    int pos = row_idx * neuron_num + tid;
    float my_val = NEG_INF;

    if(tid < neuron_num && pos < input_size)
    {
        my_val = X[pos];
        s_data[tid][0] = my_val;
    }
    else
    {
        s_data[tid][0] = NEG_INF;
    } 

    __syncthreads();

    for(int i = NUM_NEURON_ROUND/2; i > 0; i /= 2)
    {
        if(tid < i) s_data[tid][0] = fmax(s_data[tid][0], s_data[tid + i][0]);
        __syncthreads();
    }

    float max_val = s_data[0][0];
    float my_exp = 0.0f;

    if(tid < neuron_num)
    {
        my_exp = expf(my_val - max_val);
        s_data[tid][1] = my_exp;
    }
    else
    {
        s_data[tid][1] = 0.0f;
    }

    __syncthreads();

    for(int i = NUM_NEURON_ROUND/2; i >= 1; i /= 2)
    {
        if(tid < i) s_data[tid][1] += s_data[tid + i][1];
        __syncthreads();
    }

    float sum_val = s_data[0][1];

    if(tid < neuron_num && pos < input_size)
    {
        X[pos] = my_exp / sum_val;
    }

}

#define NUM_BATCH_ROUND 64
extern "C"
__global__ void loss_kernel(
    const float* __restrict__ X, 
    const int* __restrict__ L, 
    float* Y,
    int batch_size, 
    int neuron_num)
{

    __shared__ float s_data[NUM_BATCH_ROUND];


    int thread_idx = threadIdx.x;
    int row_idx = thread_idx * neuron_num;

    if(thread_idx < batch_size)
    {
        int label_idx = (int)L[thread_idx];
        float prob = X[row_idx + label_idx];
        s_data[thread_idx] = -logf(prob + 1e-9f);
    }
    else
    {
        s_data[thread_idx] = 0.0f;
    }

    __syncthreads(); 

    for(int i = NUM_BATCH_ROUND/2; i > 0; i /= 2)
    {
        if(thread_idx < i) s_data[thread_idx] += s_data[thread_idx + i];
        __syncthreads();
    }

    if(thread_idx == 0)
    {
        Y[0] = s_data[0]/(float)batch_size;
        // printf("%f", Y[0]);
    }

}

extern "C"
__global__ void softmax_kernel_backward(
    const float* __restrict__ X, 
    const int* __restrict__ L, 
    float* __restrict__ Y,
    int batch_size, 
    int neuron_num)
{
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if(block_idx < batch_size && thread_idx < neuron_num)
    {
        int pos = block_idx * neuron_num + thread_idx;
        int label_idx = L[block_idx];
    
        if(thread_idx == label_idx)
        {
            Y[pos] = (X[pos] - 1.0f) / (float)batch_size;
        }
        else
        {
            Y[pos] = X[pos] / (float)batch_size;
        }
    }


}

extern "C"
__global__ void linear_bias_backward_kernel(const float* __restrict__ dZ, float* dB, int batch_size, int neuron_num)
{
    __shared__ float s_data[NUM_BATCH_ROUND];

    int block_idx = blockIdx.x; //which col ?
    int thread_idx = threadIdx.x; //max == NUM_BATCH_ROUND (64), which row of that colum?

    int row = thread_idx * neuron_num;
    int pos = row + block_idx;

    if(thread_idx < batch_size)
    {
        s_data[thread_idx] = dZ[pos];
    }
    else
    {
        s_data[thread_idx] = 0.0f;
    }
    __syncthreads();
    for(int i = NUM_BATCH_ROUND/2; i > 0; i /= 2)
    {
        if(thread_idx < i) s_data[thread_idx] += s_data[thread_idx  + i];
        __syncthreads();
    }

    if(thread_idx == 0)
    {
        dB[block_idx] = s_data[0];
    }
}

extern "C"
__global__ void linear_weight_backward_kernel(
    const float* __restrict__ X,  // Input: [Batch, In_features]
    const float* __restrict__ dZ, // Gradient: [Batch, Out_features]
    float* __restrict__ dW,       // Output: [In_features, Out_features]
    int batch_size,
    int in_size,
    int out_size
)
{
   
    __shared__ float s_X[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_dZ[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int row_out = by * BLOCK_SIZE + ty;
    int col_out = bx * BLOCK_SIZE + tx;

    float acc = 0.0f;

    for (int k = 0; k < batch_size; k += BLOCK_SIZE)
    {
        int x_row = k + ty;
        int x_col = by * BLOCK_SIZE + tx;
        if (x_row < batch_size && x_col < in_size) {

            s_X[ty][tx] = X[x_row * in_size + x_col]; 
        } else {
            s_X[ty][tx] = 0.0f;
        }

        int dz_row = k + ty;
        int dz_col = bx * BLOCK_SIZE + tx;
        if (dz_row < batch_size && dz_col < out_size) {

            s_dZ[ty][tx] = dZ[dz_row * out_size + dz_col];
        } else {
            s_dZ[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int step = 0; step < BLOCK_SIZE; ++step)
        {
            //acc += s_X[ty][step] * s_dZ[step][tx]; but because X^T
            // => s_X[step][ty]
            acc += s_X[step][ty] * s_dZ[step][tx];
        }

        __syncthreads();
    }

    if (row_out < in_size && col_out < out_size)
    {
        dW[row_out * out_size + col_out] = acc;
    }
}

extern "C"
__global__ void linear_input_backward_kernel(
    const float* __restrict__ dZ, //[Batch, Out_features]
    const float* __restrict__ W,  //In_features, Out_features]
    float* __restrict__ dX,       //[Batch, In_features]
    int batch_size,
    int in_size,
    int out_size
)
{

    __shared__ float s_dZ[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_W[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int row_out = by * BLOCK_SIZE + ty; 
    int col_out = bx * BLOCK_SIZE + tx; 

    float acc = 0.0f;

    for (int k = 0; k < out_size; k += BLOCK_SIZE)
    {

        int dz_row = row_out; 
        int dz_col = k + tx;
        if (dz_row < batch_size && dz_col < out_size) {
            s_dZ[ty][tx] = dZ[dz_row * out_size + dz_col];
        } else {
            s_dZ[ty][tx] = 0.0f;
        }

        int w_row = bx * BLOCK_SIZE + ty; 
        int w_col = k + tx;
        if (w_row < in_size && w_col < out_size) {
            s_W[ty][tx] = W[w_row * out_size + w_col];
        } else {
            s_W[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int step = 0; step < BLOCK_SIZE; ++step)
        {
            acc += s_dZ[ty][step] * s_W[tx][step];
        }

        __syncthreads();
    }

    if (row_out < batch_size && col_out < in_size)
    {
        dX[row_out * in_size + col_out] = acc;
    }
}

extern "C"
__global__ void dropout_init_kernel(int seed, curandState_t* states, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) 
    {
        curand_init(seed, tid, 0, &states[tid]); 
    }
}

extern "C"
__global__ void dropout_forward_kernel(float* X, float* mask, curandState_t* states, float drop_rate, int size, bool is_training) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) 
    {
        if (is_training) 
        {
            
            float rand_val = curand_uniform(&states[tid]); 
            float keep_prob = 1.0f - drop_rate;
            
            
            float m = (rand_val <= keep_prob) ? (1.0f / keep_prob) : 0.0f;
            
            mask[tid] = m; 
            X[tid] *= m;
        } 
        else 
        {
            mask[tid] = 1.0f; 
        }
    }
}

extern "C"
__global__ void dropout_backward_kernel(float* d_cache, const float* mask, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) 
    {
        d_cache[tid] *= mask[tid]; 
    }
}

extern "C"
__global__ void sgd_update_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float learning_rate,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        param[idx] = param[idx] - (learning_rate * grad[idx]);
    }
}

extern "C"
__global__ void sgd_momentum_update_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ velocity,
    float learning_rate,
    float momentum,
    float weight_decay,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float current_grad = grad[idx] + (weight_decay * param[idx]);
        float v = (momentum * velocity[idx]) + (learning_rate * current_grad);
        
        velocity[idx] = v;
        
        param[idx] -= v;
    }
}

extern "C"
__global__ void adam_update_kernel(
    float* __restrict__ param, const float* __restrict__ grad,
    float* __restrict__ m, float* __restrict__ v,
    float learning_rate, float beta1, float beta2, float epsilon, float weight_decay,
    int t, int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float g = grad[idx] + (weight_decay * param[idx]);
        
        m[idx] = (beta1 * m[idx]) + (1.0f - beta1) * g;

        v[idx] = (beta2 * v[idx]) + (1.0f - beta2) * g * g;

        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));

        param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Not optimized yet
extern "C"
__global__ void conv_weight_backward_naive(
    const float* __restrict__ X,   // Input From Forward: [N, C_in, H_in, W_in]
    const float* __restrict__ dZ,  // Gradient: [N, C_out, H_out, W_out]
    float* __restrict__ dW,        // Output: [C_out, C_in, 3, 3]
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out
)
{

    int m = blockIdx.x; 
    int c = blockIdx.y; 

    int i = threadIdx.y; 
    int j = threadIdx.x; 

    if (m < C_out && c < C_in && i < 3 && j < 3)
    {
        float acc = 0.0f;

        for (int n = 0; n < N; ++n)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int in_row = h + i;
                    int in_col = w + j;

                    int dz_idx = n * (C_out * H_out * W_out) + m * (H_out * W_out) + h * W_out + w;
                    int x_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_row * W_in + in_col;

                    acc += dZ[dz_idx] * X[x_idx];
                }
            }
        }

        int dw_idx = m * (C_in * 9) + c * 9 + i * 3 + j;
        dW[dw_idx] = acc;
    }
}

// Not optimized yet
extern "C"
__global__ void conv_bias_backward_naive(
    const float* __restrict__ dZ,  // Gradient truyền về: [N, C_out, H_out, W_out]
    float* __restrict__ dB,        // Output tính ra: [C_out]
    int N, 
    int C_out, 
    int H_out, 
    int W_out
)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C_out)
    {
        float sum = 0.0f;

        for (int n = 0; n < N; ++n)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int dz_idx = n * (C_out * H_out * W_out) + c * (H_out * W_out) + h * W_out + w;
                    
                    sum += dZ[dz_idx];
                }
            }
        }

        dB[c] = sum;
    }
}

// Not optimized yet
extern "C"
__global__ void conv_input_backward_naive(
    const float* __restrict__ dZ,  // Gradient To: [N, C_out, H_out, W_out]
    const float* __restrict__ W,   // Filter: [C_out, C_in, 3, 3]
    float* __restrict__ dX,        // Gradient Go: [N, C_in, H_in, W_in]
    int N, 
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out
)
{
    int n = blockIdx.z; 
    int c_in = blockIdx.y; 

    int grid_w = (W_in + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int base_row = (blockIdx.x / grid_w) * BLOCK_SIZE;
    int base_col = (blockIdx.x % grid_w) * BLOCK_SIZE;

    int h_in = base_row + threadIdx.y;
    int w_in = base_col + threadIdx.x;

    if (h_in < H_in && w_in < W_in)
    {
        float acc = 0.0f;

        for (int c_out = 0; c_out < C_out; ++c_out)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int h_out = h_in - i;
                    int w_out = w_in - j;

                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out)
                    {
                        int dz_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                        int w_idx = c_out * (C_in * 9) + c_in * 9 + i * 3 + j;
                        
                        acc += dZ[dz_idx] * W[w_idx];
                    }
                }
            }
        }

        int dx_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in;
        dX[dx_idx] = acc;
    }
}





