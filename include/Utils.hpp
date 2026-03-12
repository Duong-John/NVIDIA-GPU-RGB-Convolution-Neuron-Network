#ifndef UTILS_HPP
#define UTILS_HPP
#include "Header.h"

void init_weight_bias(const std::string& file_name, int num_weight, int num_bias, bool is_relu);
void convert_log_to_json(const std::string& input_filepath, const std::string& output_filepath);

#endif