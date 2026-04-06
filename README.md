# Convolution Neural Network (NVIDIA GPU Implementation) on Windows - RTX 5080 Edition
This is a small, simple, solo, and personal project, which I attempt to gain insight into some features of coding on a GPU (NVIDIA GPU specifically), and try to reimplement the renowned Convolution Neural Network. The origin of this project is another project that I reimplemented the whole structure of the traditional Convolution Neuron Network using C++ Xtensor. This latest version is just the transformation from sequential execution to parallel calculation.

> **Note**: This specific branch (`rtx5080-special-edition`) was tailored, built, and tested on a High-End Desktop NVIDIA GPU (**RTX 5080 - 16GB VRAM**) paired with an AMD Ryzen 7. CUDA Version: **13.2**.

## Main Feature:
* **The current version supports some layers**: Convolution, Max Pooling, Linear, ReLU (Leaky ReLU), Softmax, Dropout (with input percentage).

* **Dataset link**: https://www.kaggle.com/c/cifar-10

- When starting, the program automatically creates and shuffles the Dataset from the folder /Dataset (CIFAR-10) with 10 classes. The dataset contains 60.000 images, of which 50.000 are training images and the rest are testing images.

* **Checkpoints**: Weights and Biases are stored in binary files in the folder /Checkpoints, and they are automatically saved and written into the files every time the latest Validation Loss is improved.

* **Early Stopping**: A limit of 20 is set so that if 20 epochs have passed since the last Weights and Biases file saving, the Model is stopped to prevent power consumption and potential overfitting. This is based on the Loss of the Validation test occurring in every epoch after the training process.

* **Data Augmentation**: The Dataset also includes some state-of-the-art Data Augmentation, such as Horizontal Flip, Pixel Shift, or blackening an area.

* **Mathematical basis**: During the process of working on the project, manual mathematical calculations were done to ensure the correctness of the theoretical fundamentals, and only then was the process of coding started.

* **Visualization**: The project is also composed of some utility stuff and a Python program to "plot" the result, and C++ SDL2 for the visualization of every kernel included in each convolution layer.

## Technical Feature:
* **Build from Scratch**: Re-implemention of every action in multi-threading in C++ and parallel computing in CUDA C++.

* **MSVC & NVCC Integration**: Shifted from MinGW (g++) to MSVC (Visual C++) toolchain to resolve CUDA 13.2 conflicts. `nvcc` is now used as the compiler driver for all `.cpp` and `.cu` files, fully utilizing the Desktop hardware capabilities.

* **Extreme Performance Leap**: Thanks to the Ryzen 7 CPU handling file decoding and the RTX 5080 executing matrix multiplications, the runtime per batch dropped from ~18ms to **8-9ms**. The training time per epoch is now reduced to approximately **6-7 seconds** (after OS page caching in the first epoch).

* **Get the batch for the next round**: Future batch fetching (`std::async`, `std::future`) has been attached in order to get the batch of the next round of training asynchronously, preventing Data Starvation for the 16GB VRAM GPU. 

* **Prevent data copying**: This project endeavors to keep the data on the VRAM of the GPU as long as possible to prevent data copying between Host (RAM & CPU) and Device (VRAM & GPU) due to the technical limitations of the throughput of the procedure.

* **Streaming**: a loop iterates through each layer in order to update Weights and Biases, which is enhanced by using Streaming on the GPU.

* **Visualization**: using pure SDL2 (VC Version) to create many Windows to visualize every kernel in the Model.

* **Memory leak on RAM & VRAM**: Ensure deallocation in VRAM after the model terminates to prevent potential memory leaks.

## How to run:
0. **Important Reminder**: This version of CNN only runs on Windows, not the WSL subsystem or Linux. It requires the **Visual Studio (MSVC)** toolchain and the **SDL2 VC Binaries**.
1. Check if your device has `make` on PowerShell or Command Prompt:
   ```bash
    make --version
    ```
   If you haven't installed `make` yet, you can use `choco` or `scoop` , or try this: https://stackoverflow.com/questions/32127524/how-can-i-install-and-use-make-in-windows

2. Choose your desired option by altering defines in `Program.cpp`:

| Define | Description |
| :--- | :--- |
| `#define READY` | Must be defined to start the Singleton Driver in order to fetch `conv_kernel.ptx` |
| `#define RUN` | Add Layers to the Model (Must have if training or running on the test dataset or single image prediction) |
| `#define INIT` | Initialize Model's parameter (Must disable when training or running on the test dataset or single image prediction) |
| `#define TRAIN` | Start training the Model on the Train Dataset. |
| `#define TEST` | Run on the Test Dataset. |
| `#define PREDICT` | Predict the label and visualize kernels of a single input image. |

3. Run the `Makefile` to compile the project. In this special edition, `nvcc` takes care of both `.cpp` and `.cu` files:
  ```bash
    make all
   ```
You can delete all the 'object' files:
```bash
    make clean
   ```
