#ifndef HEADER_H
#define HEADER_H

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <filesystem>
#include <stdexcept>
#include <future>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
//Error squiggles may occur but it's because the code editor find the 
//path of CUDA_PATH in the environment or it does understand .cu library in .cpp file

// #include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
// #include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda.h"

#include "../SDL/include/SDL.h"
#include "../SDL/include/SDL_image.h"

// #include "SDL.h"
// #include "SDL_image.h"
#include "xtensor_lib.h"

#endif