CXX = g++
NVCC = nvcc

# PATHS & FLAGS
CUDA_PATH ?= C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.9
CXXFLAGS = -w -ISDL/include -Iinclude -Itensor -I"$(CUDA_PATH)/include" -std=c++20
LDFLAGS  = -LSDL/lib1 -LSDL/lib2 -L"$(CUDA_PATH)/lib/x64" -lSDL2 -lSDL2_image -lcudart -lcuda

NVCCFLAGS = -w -arch=sm_89 -ptx

# FILES & DIRECTORIES
BUILD_DIR = build

SRCS = src/Dataset.cpp src/Convolution.cpp src/ReLU.cpp src/Max_Pooling.cpp \
       src/Utils.cpp src/Linear.cpp src/Softmax.cpp src/Loss.cpp \
       src/Optimizer.cpp src/Layer.cpp src/Model.cpp src/Dropout.cpp src/Program.cpp \
       src/Drive_Singleton.cpp

OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# File CUDA
PTX_FILE = kernel/conv_kernel.ptx
CU_SRC   = kernel/kernel.cu


# BUILD RULES
all: program $(PTX_FILE)

# CUDA (.cu) -> (.ptx)
$(PTX_FILE): $(CU_SRC)
	@echo "=> Compiling file CUDA: $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@echo "=> Compiling C++: $<"
	@mkdir -p $(BUILD_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Link file .o 
program: $(OBJS)
	@echo "=> Linking..."
	$(CXX) $(OBJS) -o program $(LDFLAGS)
	@echo "=> Complete Linking!"

# UTILITIES
clean:
	@echo "=> Cleaning project..."
	rm -rf $(BUILD_DIR) program $(PTX_FILE)