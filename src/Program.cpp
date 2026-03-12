#include "Program.hpp"
#undef main
// #define READY
// #define RUN
// #define INIT
// #define TRAIN
// #define TEST
#define RECORD
int main()
{
#ifdef READY
    try {
        Driver_Singleton::getInstance()->init("../kernel/conv_kernel.ptx");

    } catch (const std::exception& e) {
        std::cerr << "Initialization Error: " << e.what() << std::endl;
        return -1;
    }

    // In kernel.cu, it only supports batch_size <= 64
    int batch_size = 64;

    // Latest version -----------------------------------------------------------------------------------------------------------------------
    //[Scheduler] Learning Rate decayed to: 0.000004
    //[Epoch 128/200] T-Loss: 0.7009 | T-Acc: 75.42% || V-Loss: 0.8598 | V-Acc: 71.80% | Time: 16697 ms --> [Early Stopping] Patience: 20/20
    //[Test Results] Total Samples: 10000 | Loss: 0.7810 | Accuracy: 73.65% | Time: 1287 ms
    Convolution conv1(3, 64, 32, 32, "Checkpoints/convolution_kernel_1.bin");
    ReLU relu_conv1(batch_size, 64, 30, 30);
    Max_Pooling pool1(batch_size, 64, 30, 30);


    Convolution conv2(64, 128, 15, 15, "Checkpoints/convolution_kernel_2.bin");
    ReLU relu_conv2(batch_size, 128, 13, 13);
    Max_Pooling pool2(batch_size, 128, 13, 13);


    // 128 * 7 * 7 = 6272
    Linear lin1(batch_size, 6272, 512, "Checkpoints/linear_1.bin");
    ReLU relu1(batch_size, 512, 1, 1);

    Dropout drop1(batch_size, 512, 0.35f); 
    
    Linear lin2(batch_size, 512, 128, "Checkpoints/linear_2.bin");
    ReLU relu2(batch_size, 128, 1, 1);
    Dropout drop2(batch_size, 128, 0.15f); 
    
    Linear lin3(batch_size, 128, 10, "Checkpoints/linear_3.bin");
    Softmax softmax(batch_size, 10, 1, 1);

    Loss loss(batch_size, 10, 1, 1);
    // Latest version -----------------------------------------------------------------------------------------------------------------------

    // 70% Accuracy -----------------------------------------------------------
    // Convolution conv1(3, 32, 32, 32, "Checkpoints/convolution_kernel_1.bin");
    // ReLU relu_conv1(batch_size, 32, 30, 30);
    // Max_Pooling pool1(batch_size, 32, 30, 30);

    // Convolution conv2(32, 64, 15, 15, "Checkpoints/convolution_kernel_2.bin");
    // ReLU relu_conv2(batch_size, 64, 13, 13);
    // Max_Pooling pool2(batch_size, 64, 13, 13);

    // // 64 * 7 * 7 = 3136
    // Linear lin1(batch_size, 3136, 256, "Checkpoints/linear_1.bin");
    // ReLU relu1(batch_size, 256, 1, 1);
    // Dropout drop1(batch_size, 256, 0.25f);
    
    // Linear lin2(batch_size, 256, 64, "Checkpoints/linear_2.bin");
    // ReLU relu2(batch_size, 64, 1, 1);
    // Dropout drop2(batch_size, 64, 0.1f);
    
    // Linear lin3(batch_size, 64, 10, "Checkpoints/linear_3.bin");
    // Softmax softmax(batch_size, 10, 1, 1);

    // Loss loss(batch_size, 10, 1, 1);
    // 70% Accuracy -----------------------------------------------------------

#endif

#ifdef INIT
    // init_weight_bias("Checkpoints/linear_1.bin", 3136*256, 256, true);
    // init_weight_bias("Checkpoints/linear_2.bin", 256*64, 64, true);
    // init_weight_bias("Checkpoints/linear_3.bin", 64*10, 10, false);
    // conv1.init_weights("Checkpoints/convolution_kernel_1.bin");
    // conv2.init_weights("Checkpoints/convolution_kernel_2.bin");

    init_weight_bias("Checkpoints/linear_1.bin", 6272 * 512, 512, true);
    init_weight_bias("Checkpoints/linear_2.bin", 512 * 128, 128, true);
    init_weight_bias("Checkpoints/linear_3.bin", 128 * 10, 10, false);
    
    conv1.init_weights("Checkpoints/convolution_kernel_1.bin");
    conv2.init_weights("Checkpoints/convolution_kernel_2.bin");

    return 0;
    
#endif
    
#ifdef RUN
    //In kernel.cu, it only support number of class <= 32
    //Model(const std::string& dataset_path, int batch_size, int num_classes, float learning_rate, OptimizerType opt_type, float momentum = 0.9f, float weight_decay = 0.0001f);
    Model my_model("Dataset/Dataset_10/train", batch_size, 10, 0.001f, OptimizerType::ADAM, 0.9f, 0.0001f);
    

    // Best use -----------------------------------------------------------
    my_model.add_layer(&conv1);
    my_model.add_layer(&relu_conv1);
    my_model.add_layer(&pool1);

    my_model.add_layer(&conv2);
    my_model.add_layer(&relu_conv2);
    my_model.add_layer(&pool2);

    my_model.add_layer(&lin1);
    my_model.add_layer(&relu1);
    my_model.add_layer(&drop1);

    my_model.add_layer(&lin2);
    my_model.add_layer(&relu2);
    my_model.add_layer(&drop2);

    my_model.add_layer(&lin3);
    my_model.add_layer(&softmax);
    
    my_model.set_loss(&loss);
    // Best use -----------------------------------------------------------
#endif

#ifdef TRAIN
    std::cout << "\n--- STARTING TRAINING ON TRAINING SET ---\n";
    int epochs;
    std::cout << "[Program] Enter epochs: "; std::cin >> epochs;
    
    auto start = std::chrono::high_resolution_clock::now();
    my_model.train(epochs);

    auto end = std::chrono::high_resolution_clock::now();
    int elapsed_ms = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "\r\033[K\033[1;32mTime: " << elapsed_ms << " s\033[0m" << std::endl;
    // my_model.save_model();

#endif


#ifdef TEST
    std::cout << "\n--- STARTING EVALUATION ON TEST SET ---\n";
    my_model.test("Dataset/Dataset_10/test");
#endif

#ifdef RECORD
    convert_log_to_json("report/Report 12.3.2026.txt", "report/log.json");

#endif

    return 0;

}