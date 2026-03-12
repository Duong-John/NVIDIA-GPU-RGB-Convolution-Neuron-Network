#include "Drive_Singleton.hpp"

Driver_Singleton* Driver_Singleton::instance = nullptr;

Driver_Singleton* Driver_Singleton::getInstance() {
    if (instance == nullptr) {
        instance = new Driver_Singleton();
    }
    return instance;
}

void Driver_Singleton::init(const char* ptx_path) {
    if (is_initialized) return;

    //Init CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) throw std::runtime_error("[Drive_Singleton] Failed to init CUDA Driver API");

    //Get Device(GPU 0)
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) throw std::runtime_error("[Drive_Singleton] Failed to get CUDA Device");

    //Create Context
    res = cuCtxCreate(&context, 0, device);
    if (res != CUDA_SUCCESS) throw std::runtime_error("[Drive_Singleton] Failed to create CUDA Context");

    //Load PTX to module
    res = cuModuleLoad(&module, ptx_path);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("[Drive_Singleton] Failed to load PTX file: ") + ptx_path);
    }

    std::cout << "[Drive_Singleton] CUDA Initialized. Loaded PTX: " << ptx_path << std::endl;
    is_initialized = true;
}

CUmodule Driver_Singleton::getModule() const {
    return module;
}

CUcontext Driver_Singleton::getContext() const {
    return context;
}

Driver_Singleton::~Driver_Singleton() {
    if (is_initialized) {
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }
}