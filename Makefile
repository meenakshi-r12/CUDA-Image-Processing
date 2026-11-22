
NVCC = /usr/local/cuda/bin/nvcc
TARGET = image_processor
SOURCE = src/image_processing.cu
NVCC_FLAGS = -O2 -std=c++14 -arch=compute_80 -code=compute_80

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -lpng -o $(TARGET)

clean:
	rm -f $(TARGET)
