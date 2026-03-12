#include "Dataset.hpp"

Dataset::Dataset(const std::string& root_dir, bool train, float val_split) 
{
    this->is_train = train;

    if (!(IMG_Init(IMG_INIT_PNG) & IMG_INIT_PNG)) {
        std::cerr << "SDL_image Init Error: " << IMG_GetError() << std::endl;
    }
    
    std::cout << "[Dataset] Scanning directory: " << root_dir << "..." << std::endl;

    std::vector<std::string> temp_filepaths;
    std::vector<int> temp_labels;

    scan_directories(root_dir, temp_filepaths, temp_labels);

    if (this->is_train) 
    {
        std::random_device rd;
        std::mt19937 g(12345);
        for(size_t i = temp_filepaths.size() - 1; i > 0; i--) {

            std::uniform_int_distribution<size_t> dist(0, i);
            size_t j = dist(g);
            std::swap(temp_filepaths[i], temp_filepaths[j]);

            std::swap(temp_labels[i], temp_labels[j]);
        }

        size_t val_size = static_cast<size_t>(temp_filepaths.size() * val_split);
        size_t train_size = temp_filepaths.size() - val_size;

        train_filepaths.assign(temp_filepaths.begin(), temp_filepaths.begin() + train_size);

        train_labels.assign(temp_labels.begin(), temp_labels.begin() + train_size);

        val_filepaths.assign(temp_filepaths.begin() + train_size, temp_filepaths.end());

        val_labels.assign(temp_labels.begin() + train_size, temp_labels.end());

        std::cout << "[Dataset] Split -> Train: " << train_size << " images | Validation: " << val_size << " images." << std::endl;
    }
    else 
    {
        train_filepaths = temp_filepaths;
        train_labels = temp_labels;
        std::cout << "[Dataset] Test Mode -> Loaded " << train_filepaths.size() << " images." << std::endl;
    }
}

Dataset::~Dataset()
{
    IMG_Quit();
}

size_t Dataset::get_train_size() const 
{ 
    return train_filepaths.size(); 
}

size_t Dataset::get_val_size() const 
{ 
    return val_filepaths.size(); 
}

// size_t Dataset::size() const 
// {
//     return all_filepaths.size();
// }

void Dataset::shuffle()
{
    if (!is_train) return; 

    std::random_device rd;
    std::mt19937 g(rd());

    for(size_t i = train_filepaths.size() - 1; i > 0; i--)
    {
        std::uniform_int_distribution<size_t> dist(0, i);

        size_t j = dist(g);
        std::swap(train_filepaths[i], train_filepaths[j]);

        std::swap(train_labels[i], train_labels[j]);
    }
}

void Dataset::scan_directories(const std::string& root_dir, std::vector<std::string>& temp_filepaths, std::vector<int>& temp_labels)
{
    if(!std::filesystem::exists(root_dir)) throw std::runtime_error("Dataset Directory not found: " + root_dir);

    int label_index = 0;
    for(const auto& entry: std::filesystem::directory_iterator(root_dir))
    {
        if(entry.is_directory()) class_names.push_back(entry.path().filename().string());

        for(const auto& img_entry: std::filesystem::directory_iterator(entry.path()))
        {
            if(img_entry.path().extension() == ".png") {

                temp_filepaths.push_back(img_entry.path().string());
                temp_labels.push_back(label_index);
            }
        }
        label_index++;
    }
}

void Dataset::load_chunk_worker(size_t start_index, size_t end_index, const std::vector<std::string>& filepaths, float*data_ptr, bool apply_augmentation)
{
    const size_t image_size = C*W*H;
    const size_t area = H*W; //1 channel

    std::mt19937 gen(std::random_device{}() + start_index);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_shift(-2, 2);
    std::uniform_int_distribution<int> dist_cutout(0, W - 8);

    for(size_t i = start_index; i < end_index; ++i)
    {
        SDL_Surface* surface = IMG_Load(filepaths[i].c_str());
        if(!surface)
        {
            SDL_Log("Error loading image: %s", IMG_GetError());
            std::cout<<"Error loading image: "<<filepaths[i]<<std::endl;
            continue;
        }
    
        SDL_Surface* fmt_surface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBX8888, 0);
        SDL_FreeSurface(surface);

        Uint32* pixels = (Uint32*)fmt_surface->pixels;

        float* current_image_ptr = data_ptr + (i - start_index) * image_size;
        std::fill(current_image_ptr, current_image_ptr + image_size, 0.0f); //Blacken the image first

        bool do_flip = apply_augmentation && (dist(gen) > 0.5f);
        int dx = apply_augmentation ? dist_shift(gen) : 0;
        int dy = apply_augmentation ? dist_shift(gen) : 0;

        bool do_cutout = apply_augmentation && (dist(gen) > 0.5f);
        int cut_x = dist_cutout(gen);
        int cut_y = dist_cutout(gen);
        
        for(int h = 0; h < H; ++h)
        {
            for(int w = 0; w < W; ++w)
            {
                if(h >= fmt_surface->h || w >= fmt_surface->w) continue;

                int h_out = h + dy;
                int w_out = (do_flip ? (W - 1 - w) : w) + dx;

                bool in_cutout_box = do_cutout && (h_out >= cut_y && h_out < cut_y + 8) && (w_out >= cut_x && w_out < cut_x + 8);

                if (h_out >= 0 && h_out < H && w_out >= 0 && w_out < W && !in_cutout_box)
                {
                    Uint32 pixel = pixels[h * fmt_surface->w + w];
                    Uint8 r,g,b,a;
                    SDL_GetRGBA(pixel, fmt_surface->format, &r, &g, &b, &a);

                    // int w_out = do_flip ? (W - 1 - w) : w;

                    // current_image_ptr[0*area + h*W + w] = static_cast<float>(r) / 255.0f;
                    // current_image_ptr[1*area + h*W + w] = static_cast<float>(g) / 255.0f;
                    // current_image_ptr[2*area + h*W + w] = static_cast<float>(b) / 255.0f;
                    current_image_ptr[0*area + h_out*W + w_out] = static_cast<float>(r) / 255.0f;
                    current_image_ptr[1*area + h_out*W + w_out] = static_cast<float>(g) / 255.0f;
                    current_image_ptr[2*area + h_out*W + w_out] = static_cast<float>(b) / 255.0f;
                }
            }
        }

        SDL_FreeSurface(fmt_surface);
    }
}

std::pair<xt::xarray<float>, xt::xarray<int>> Dataset::get_batch(size_t start_index, size_t batch_size, bool is_val)
{
    const std::vector<std::string>& current_filepaths = is_val ? val_filepaths : train_filepaths;

    const std::vector<int>& current_labels = is_val ? val_labels : train_labels;

    size_t end_index = std::min(start_index + batch_size, current_filepaths.size());
    size_t real_batch_size = end_index - start_index;

    xt::xarray<float> batch_data = xt::empty<float>({real_batch_size, (size_t)C, (size_t)H, (size_t)W});

    xt::xarray<float> batch_labels = xt::empty<int>({real_batch_size});

    float* raw_data_ptr = batch_data.data();
    unsigned int num_threads = std::thread::hardware_concurrency();

    if(num_threads == 0) num_threads = 4;
    if(num_threads > real_batch_size) num_threads = real_batch_size;

    std::vector<std::thread> workers;
    size_t items_per_thread = (real_batch_size + num_threads - 1) / num_threads; 

    bool apply_aug = this->is_train && !is_val;

    for(unsigned int t = 0; t < num_threads; ++t)
    {
        size_t t_start = start_index + t * items_per_thread;

        size_t t_end = std::min(t_start + items_per_thread, end_index);

        if(t_start >= t_end) break;

        size_t offset_in_batch = (t_start - start_index) * (C*H*W);
        
        workers.emplace_back(&Dataset::load_chunk_worker, this, t_start, t_end, std::cref(current_filepaths), raw_data_ptr + offset_in_batch, apply_aug);
    }

    for(size_t i = 0; i < real_batch_size; ++i) 
        batch_labels(i) = current_labels[start_index + i];

    for(auto& t : workers) 
        if(t.joinable()) 
            t.join();

    return {batch_data, batch_labels};
}