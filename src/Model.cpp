#include "Model.hpp"
#include <chrono>

// Model::Model(const std::string& dataset_path, int batch_size, int num_classes, float learning_rate)
//     : dataset_path(dataset_path), batch_size(batch_size), num_classes(num_classes), optimizer(learning_rate)
// {
// }

// Model::Model(const std::string &dataset_path, int batch_size, int num_classes, float learning_rate, float momentum, float weight_decay)
//     : dataset_path(dataset_path), batch_size(batch_size), num_classes(num_classes), optimizer(learning_rate, momentum, weight_decay)
// {
// }

Model::Model(const std::string &dataset_path, int batch_size, int num_classes, float learning_rate, OptimizerType opt_type, float momentum, float weight_decay)
    : dataset_path(dataset_path), batch_size(batch_size), num_classes(num_classes)
{
    this->opt_type = opt_type;

    if (opt_type == OptimizerType::SGD) {
        this->optimizer = new SGD(learning_rate);
    } 
    else if (opt_type == OptimizerType::SGD_MOMENTUM) {
        this->optimizer = new SGD_Momentum(learning_rate, momentum, weight_decay);
    } 
    else if (opt_type == OptimizerType::ADAM) {
        this->optimizer = new Adam(learning_rate, 0.9f, 0.999f, 1e-8f, weight_decay);
    }
}

Model::~Model()
{
    for (CUstream& stream : streams) {
        cuStreamDestroy(stream);
    }
    if (d_input_buffer) {
        cudaFree(d_input_buffer);
    }
}

void Model::add_layer(Layer* in_layer)
{
    pipeline.push_back(in_layer);

    CUstream stream_w, stream_b;

    cuStreamCreate(&stream_w, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream_b, CU_STREAM_NON_BLOCKING);
    streams.push_back(stream_w);
    streams.push_back(stream_b);
}

void Model::set_loss(Loss* in_loss)
{
    this->loss_layer = in_loss;
}

void Model::train(int epochs)
{
    if (loss_layer == nullptr) throw std::runtime_error("Loss layer is not set!");

    Dataset dataset(dataset_path, true, 0.2f); 
    int train_size = dataset.get_train_size();
    int val_size = dataset.get_val_size();


    float* h_output = new float[batch_size * num_classes];

    for (Layer* layer : pipeline)
    {
        layer->load_weight_bias();
        layer->is_training = true;
        layer->set_optimizer_type(this->opt_type);
    } 

    float best_val_loss = 1e9f;
    int wait = 0;
    int wait_limit = 20;

    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        //Scheduler
        if (epoch > 0 && epoch % 15 == 0) 
        {
            float current_lr = this->optimizer->get_learning_rate();
            float new_lr = current_lr * 0.5f;
            this->optimizer->set_learning_rate(new_lr);

            std::cout << "\n\033[1;35m[Scheduler] Learning Rate decayed to: " << std::fixed << std::setprecision(6) << new_lr << "\033[0m\n";
        }

        dataset.shuffle(); 
        auto start = std::chrono::high_resolution_clock::now();
        
        float train_loss_sum = 0.0f;
        int train_correct = 0;       
        int total_train_samples = 0; 

        for(int i = 0; i < train_size; i += batch_size)
        {
            auto X_batch = dataset.get_batch(i, batch_size, false);
            size_t last_batch = X_batch.first.shape()[0];

            size_t current_bytes = X_batch.first.size() * sizeof(float);
            if (buffer_size < current_bytes) 
            {
                if (d_input_buffer) cudaFree(d_input_buffer);

                cudaMalloc(&d_input_buffer, current_bytes);
                buffer_size = current_bytes;
            }
            cudaMemcpy(d_input_buffer, X_batch.first.data(), current_bytes, cudaMemcpyHostToDevice);

            float* output = d_input_buffer;

            for (Layer* layer : pipeline) output = layer->forward(output, last_batch);
            

            cudaMemcpy(h_output, output, last_batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
            
            for(size_t b = 0; b < last_batch; ++b) 
            {
                int best_class = 0;


                float max_prob = h_output[b * num_classes];

                for(int c = 1; c < num_classes; ++c) 
                {
                    if(h_output[b * num_classes + c] > max_prob) 
                    {
                        max_prob = h_output[b * num_classes + c];
                        best_class = c;
                    }
                }

                if(best_class == X_batch.second(b)) train_correct++;
            }
            total_train_samples += last_batch;


            output = loss_layer->forward(output, X_batch.second);
            train_loss_sum += loss_layer->get_loss();

            output = loss_layer->backward();
            for (int j = pipeline.size() - 1; j >= 0; --j) output = pipeline[j]->backward(output);

            int stream_idx = 0;
            for (Layer* layer : pipeline) 
            {
                layer->update_params(*(this->optimizer), streams[stream_idx], streams[stream_idx + 1]);
                stream_idx += 2;
            }

            cuCtxSynchronize(); 


            float progress = (float)(i + batch_size) / train_size * 100.0f;
            if (progress > 100.0f) progress = 100.0f;
            float current_acc = (float)train_correct / total_train_samples * 100.0f;

            std::cout << "\r\033[K\033[1;33m[Epoch " << epoch + 1 << "/" << epochs << "] "
                      << "Train Progress: " << std::fixed << std::setprecision(1) << progress << "% | "
                      << "Acc: " << current_acc << "%\033[0m" << std::flush;
        }

        float val_loss_sum = 0.0f;
        int val_correct = 0;       

        int total_val_samples = 0; 
        int val_batches = 0;

        for(int i = 0; i < val_size; i += batch_size)
        {
            auto X_batch = dataset.get_batch(i, batch_size, true);
            size_t last_batch = X_batch.first.shape()[0];
            size_t current_bytes = X_batch.first.size() * sizeof(float);
            
            cudaMemcpy(d_input_buffer, X_batch.first.data(), current_bytes, cudaMemcpyHostToDevice);

            float* output = d_input_buffer;
            for (Layer* layer : pipeline) output = layer->forward(output, last_batch);
            

            cudaMemcpy(h_output, output, last_batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

            for(size_t b = 0; b < last_batch; ++b) 
            {
                int best_class = 0;
                float max_prob = h_output[b * num_classes];

                for(int c = 1; c < num_classes; ++c) 
                {
                    if(h_output[b * num_classes + c] > max_prob) 
                    {
                        max_prob = h_output[b * num_classes + c];
                        best_class = c;
                    }
                }
                if(best_class == X_batch.second(b)) val_correct++;
            }
            total_val_samples += last_batch;


            output = loss_layer->forward(output, X_batch.second);
            cuCtxSynchronize(); 
            val_loss_sum += loss_layer->get_loss();
            val_batches++;


            float progress = (float)(i + batch_size) / val_size * 100.0f;
            if (progress > 100.0f) progress = 100.0f;

            float current_acc = (float)val_correct / total_val_samples * 100.0f;

            std::cout << "\r\033[K\033[1;36m[Epoch " << epoch + 1 << "/" << epochs << "] "
                      << "Val Progress: " << std::fixed << std::setprecision(1) << progress << "% | "
                      << "Acc: " << current_acc << "%\033[0m" << std::flush;
        }

    
        auto end = std::chrono::high_resolution_clock::now();
        int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        float final_train_loss = train_loss_sum / (total_train_samples / (float)batch_size);

        float final_train_acc = (float)train_correct / total_train_samples * 100.0f;
        
        float final_val_loss = (val_batches > 0) ? (val_loss_sum / val_batches) : 0.0f;
        float final_val_acc = (total_val_samples > 0) ? ((float)val_correct / total_val_samples * 100.0f) : 0.0f;


        std::cout << "\r\033[K\033[1;32m[Epoch " << epoch + 1 << "/" << epochs << "] "
                  << "T-Loss: " << std::fixed << std::setprecision(4) << final_train_loss << " | "
                  << "T-Acc: " << std::setprecision(2) << final_train_acc << "% || "
                  << "V-Loss: " << std::setprecision(4) << final_val_loss << " | "
                  << "V-Acc: " << std::setprecision(2) << final_val_acc << "% | "
                  << "Time: " << elapsed_ms << " ms\033[0m";

        if (final_val_loss < best_val_loss) 
        {
            best_val_loss = final_val_loss;
            wait = 0;

            std::cout << " --> [Early Stopping] New Best Model Saved! (V-Loss: " << best_val_loss << ")\n";
            this->save_model();
            // std::cout<<std::endl;
        }
        else 
        {
            wait++;
            std::cout << " --> [Early Stopping] Patience: " << wait << "/" << wait_limit << "\n";

            if (wait >= wait_limit) 
            {
                std::cout << "\n\033[1;31m[Early Stopping] Triggered! Training aborted to prevent overfitting.\033[0m\n";
                break;
            }
        }
        // std::cout<<std::endl;
    }

    delete[] h_output; // Deallocate RAM
}

void Model::test(const std::string& test_path)
{
    if (loss_layer == nullptr) throw std::runtime_error("Loss layer is not set!");

    Dataset test_dataset(test_path, false); 
    
    int test_size = test_dataset.get_train_size(); 

    if (test_size == 0) 
    {
        std::cout << "[Model] No images found in test path!" << std::endl;
        return;
    }

    float* h_output = new float[batch_size * num_classes];

    float test_loss_sum = 0.0f;

    int test_correct = 0;       
    int total_test_samples = 0; 
    int test_batches = 0;

    for (Layer* layer : pipeline)
    {
        layer->load_weight_bias();
        layer->is_training = false;
    } 

    auto start = std::chrono::high_resolution_clock::now();


    for(int i = 0; i < test_size; i += batch_size)
    {
        auto X_batch = test_dataset.get_batch(i, batch_size, false);
        size_t last_batch = X_batch.first.shape()[0];
        
        size_t current_bytes = X_batch.first.size() * sizeof(float);
        if (buffer_size < current_bytes) 
        {
            if (d_input_buffer) cudaFree(d_input_buffer);

            cudaMalloc(&d_input_buffer, current_bytes);
            buffer_size = current_bytes;
        }
        cudaMemcpy(d_input_buffer, X_batch.first.data(), current_bytes, cudaMemcpyHostToDevice);

        float* output = d_input_buffer;
        for (Layer* layer : pipeline) 
        {
            output = layer->forward(output, last_batch);
        }

        cudaMemcpy(h_output, output, last_batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

        for(size_t b = 0; b < last_batch; ++b) 
        {
            int best_class = 0;
            float max_prob = h_output[b * num_classes];

            for(int c = 1; c < num_classes; ++c) 
            
            {
                if(h_output[b * num_classes + c] > max_prob) 
                {
                    max_prob = h_output[b * num_classes + c];
                    best_class = c;
                }
            }
            if(best_class == X_batch.second(b)) test_correct++;
        }
        total_test_samples += last_batch;

        output = loss_layer->forward(output, X_batch.second);
        cuCtxSynchronize(); 
        test_loss_sum += loss_layer->get_loss();
        test_batches++;


        float progress = (float)(i + batch_size) / test_size * 100.0f;

        if (progress > 100.0f) progress = 100.0f;
        float current_acc = (float)test_correct / total_test_samples * 100.0f;

        std::cout << "\r\033[K\033[1;35m[Testing] "
                  << "Progress: " << std::fixed << std::setprecision(1) << progress << "% | "
                  << "Acc: " << current_acc << "%\033[0m" << std::flush;
    }


    auto end = std::chrono::high_resolution_clock::now();

    int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    float final_test_loss = (test_batches > 0) ? (test_loss_sum / test_batches) : 0.0f;
    float final_test_acc = (total_test_samples > 0) ? ((float)test_correct / total_test_samples * 100.0f) : 0.0f;


    std::cout << "\r\033[K\033[1;32m[Test Results] "
              << "Total Samples: " << total_test_samples << " | "
              << "Loss: " << std::fixed << std::setprecision(4) << final_test_loss << " | "
              << "Accuracy: " << std::setprecision(2) << final_test_acc << "% | "
              << "Time: " << elapsed_ms << " ms\033[0m" << std::endl;

    delete[] h_output;
}

void Model::save_model()
{
    for (Layer* layer : pipeline) {
        layer->save_weight_bias();
    }
    std::cout << "[Model] All model checkpoints saved successfully!" << std::endl;
}