#ifndef DRIVER_SINGLETON_HPP
#define DRIVER_SINGLETON_HPP
#include "Header.h"


class Driver_Singleton {
private:
    static Driver_Singleton* instance;
    CUdevice device;
    CUcontext context;
    CUmodule module; // code kernel (file .ptx)
    bool is_initialized = false;

    //Singleton Pattern
    Driver_Singleton() {}

public:
    static Driver_Singleton* getInstance();

    //Load file PTX
    void init(const char* ptx_path);
    

    CUmodule getModule() const;
    CUcontext getContext() const;

    ~Driver_Singleton();
};

#endif