#ifndef DATASET_HPP
#define DATASET_HPP
#include "Header.h"

class Dataset {
public:
   
    const int C = 3; 
    const int H = 32;
    const int W = 32;

    // std::vector<std::string> all_filepaths;
    // std::vector<int> all_labels;
    // std::vector<std::string> class_names;

    std::vector<std::string> train_filepaths;
    std::vector<int> train_labels;
    std::vector<std::string> val_filepaths;
    std::vector<int> val_labels;
    std::vector<std::string> class_names;


    bool is_train;

    Dataset(const std::string& root_dir, bool train = true, float val_split = 0.2f);
    ~Dataset();

    // size_t size() const;

    void shuffle();

    std::pair<xt::xarray<float>, xt::xarray<int>> get_batch(size_t start_index, size_t batch_size, bool is_val = false);
    size_t get_train_size() const;
    size_t get_val_size() const;

private:
    
    void scan_directories(const std::string& root_dir, std::vector<std::string>& temp_filepaths, std::vector<int>& temp_labels);

    void load_chunk_worker(size_t start_index, size_t end_index, 
                           const std::vector<std::string>& filepaths, 
                           float* data_ptr, bool apply_augmentation);
};

#endif