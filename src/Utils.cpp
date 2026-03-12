#include "Utils.hpp"

// void init_weight_bias(const std::string &file_name, int num_weight, int num_bias)
// {
//     std::ofstream file(file_name, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "[Utils] Cannot create file: " << file_name << std::endl;
//         return;
//     }

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

//     for (int i = 0; i < num_weight; ++i) {
//         float val = dis(gen);
//         file.write(reinterpret_cast<const char*>(&val), sizeof(float));
//     }

//     for (int i = 0; i < num_bias; ++i) {
//         float val = dis(gen);
//         file.write(reinterpret_cast<const char*>(&val), sizeof(float));
//     }

//     file.close();
//     std::cout << "[Utils] Initialized " << file_name << " with " << num_weight << " weights and " << num_bias << " biases." << std::endl;
// }

void init_weight_bias(const std::string &file_name, int num_weight, int num_bias, bool is_relu)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot create file: " << file_name << std::endl;
        return;
    }

    int fan_in = num_weight / num_bias;
    
    // ReLU -> Kaiming (2.0)
    // Softmax -> Xavier (1.0)
    float numerator = is_relu ? 2.0f : 1.0f; 
    float stddev = std::sqrt(numerator / fan_in);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, stddev);

    for (int i = 0; i < num_weight; ++i) {
        float val = dis(gen);
        file.write(reinterpret_cast<const char*>(&val), sizeof(float));
    }

    for (int i = 0; i < num_bias; ++i) {
        float val = 0.01f; 
        file.write(reinterpret_cast<const char*>(&val), sizeof(float));
    }

    file.close();
    
    std::string method = is_relu ? "Kaiming (ReLU)" : "Xavier (Softmax)";
    std::cout << "[Utils] Initialized " << file_name << " using " << method << std::endl;
}

void convert_log_to_json(const std::string& input_filepath, const std::string& output_filepath) 
{
    std::ifstream infile(input_filepath);
    if (!infile.is_open()) {
        std::cerr << "[Utils] Error: Cannot open input log file!" << std::endl;
        return;
    }

    std::ofstream outfile(output_filepath);
    if (!outfile.is_open()) {
        std::cerr << "[Utils] Error: Cannot open output JSON file!" << std::endl;
        return;
    }

    //Start JSON
    outfile << "[\n";
    
    std::string line;
    bool is_first_entry = true;

    while (std::getline(infile, line)) 
    {
        size_t epoch_start = line.find("[Epoch ");
        if (epoch_start != std::string::npos) 
        {
            size_t slash_pos = line.find("/", epoch_start);
            size_t t_loss_pos = line.find("T-Loss:");
            size_t t_acc_pos = line.find("T-Acc:");
            size_t v_loss_pos = line.find("V-Loss:");
            size_t v_acc_pos = line.find("V-Acc:");
            size_t time_pos = line.find("Time:");

            if (slash_pos != std::string::npos && t_loss_pos != std::string::npos && 
                t_acc_pos != std::string::npos && v_loss_pos != std::string::npos && 
                v_acc_pos != std::string::npos && time_pos != std::string::npos) 
            {
                std::string epoch_str = line.substr(epoch_start + 7, slash_pos - (epoch_start + 7));

                size_t t_loss_end = line.find("|", t_loss_pos);
                std::string t_loss_str = line.substr(t_loss_pos + 7, t_loss_end - (t_loss_pos + 7));

                size_t t_acc_end = line.find("%", t_acc_pos);
                std::string t_acc_str = line.substr(t_acc_pos + 6, t_acc_end - (t_acc_pos + 6));

                size_t v_loss_end = line.find("|", v_loss_pos);
                std::string v_loss_str = line.substr(v_loss_pos + 7, v_loss_end - (v_loss_pos + 7));

                size_t v_acc_end = line.find("%", v_acc_pos);
                std::string v_acc_str = line.substr(v_acc_pos + 6, v_acc_end - (v_acc_pos + 6));

                size_t time_end = line.find("ms", time_pos);
                std::string time_str = line.substr(time_pos + 5, time_end - (time_pos + 5));

                // Write into JSON
                if (!is_first_entry) 
                {
                    outfile << ",\n";
                }
                is_first_entry = false;

                outfile << "  {\n"
                        << "    \"epoch\": " << std::stoi(epoch_str) << ",\n"
                        << "    \"t_loss\": " << std::stof(t_loss_str) << ",\n"
                        << "    \"t_acc\": " << std::stof(t_acc_str) << ",\n"
                        << "    \"v_loss\": " << std::stof(v_loss_str) << ",\n"
                        << "    \"v_acc\": " << std::stof(v_acc_str) << ",\n"
                        << "    \"time_ms\": " << std::stoi(time_str) << "\n"
                        << "  }";
            }
        }
    }
    
    // End JSON
    outfile << "\n]\n";
    
    infile.close();
    outfile.close();
    std::cout << "[Utils] Log converted to JSON successfully: " << output_filepath << std::endl;
}