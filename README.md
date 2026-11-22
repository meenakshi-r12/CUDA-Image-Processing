
# GPU-Accelerated Image Processing Using CUDA

## Project Overview
This project implements batch image processing using custom CUDA C++ kernels on an NVIDIA GPU. It processes multiple PNG images and generates grayscale, Gaussian blur, and Sobel edge detection outputs.

## Technologies Used
- CUDA C++
- NVIDIA L4 GPU
- libpng
- Makefile

## CUDA Kernels
1. **Grayscale:** Converts RGB images into grayscale.
2. **Gaussian Blur:** Applies a 3x3 Gaussian filter to smooth images.
3. **Sobel Edge Detection:** Detects image edges using horizontal and vertical gradients.

## Dataset
The project uses 100 synthetic PNG images, each with a resolution of 512x512 pixels.

## Build Instructions
```bash
make
```

## Run Instructions
```bash
./image_processor --input dataset/input_images --output output
```

## Output
The program generates three folders:
- `output/grayscale`
- `output/blur`
- `output/edges`

For 100 input images, the program generates 300 processed images.

## Execution Results
- GPU: NVIDIA L4
- Input images processed: 100
- Output images generated: 300
- Recorded execution time: approximately 5–6 seconds in the Coursera lab.

## Learning Outcomes
- Understanding CUDA thread and block execution.
- Implementing custom GPU kernels.
- Managing host-to-device and device-to-host memory transfers.
- Applying parallel image-processing algorithms.
- Processing a batch of images using GPU acceleration.
