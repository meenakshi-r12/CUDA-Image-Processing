#include <cuda_runtime.h>
#include <png.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void grayscaleKernel(
    const unsigned char* input, unsigned char* gray,
    int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = (y * width + x) * 3;

        gray[y * width + x] =
            (unsigned char)(0.299f * input[i] +
                            0.587f * input[i + 1] +
                            0.114f * input[i + 2]);
    }
}

__global__ void gaussianKernel(
    const unsigned char* input, unsigned char* output,
    int width, int height) {

    const int weights[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int px = max(0, min(width - 1, x + dx));
                int py = max(0, min(height - 1, y + dy));

                sum += input[py * width + px] *
                       weights[dy + 1][dx + 1];
            }
        }

        output[y * width + x] = (unsigned char)(sum / 16);
    }
}

__global__ void sobelKernel(
    const unsigned char* input, unsigned char* output,
    int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int gy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        int sx = 0;
        int sy = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int px = max(0, min(width - 1, x + dx));
                int py = max(0, min(height - 1, y + dy));

                int value = input[py * width + px];

                sx += value * gx[dy + 1][dx + 1];
                sy += value * gy[dy + 1][dx + 1];
            }
        }

        float magnitude = sqrtf((float)(sx * sx + sy * sy));
        output[y * width + x] =
            (unsigned char)fminf(255.0f, magnitude);
    }
}

bool readPNG(const std::string& path,
             std::vector<unsigned char>& pixels,
             int& width, int& height) {

    png_image image{};
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, path.c_str())) {
        std::cerr << "Cannot read: " << path << std::endl;
        return false;
    }

    image.format = PNG_FORMAT_RGB;
    width = image.width;
    height = image.height;

    pixels.resize(PNG_IMAGE_SIZE(image));

    if (!png_image_finish_read(
            &image, nullptr, pixels.data(), 0, nullptr)) {
        std::cerr << "PNG read failed: " << path << std::endl;
        png_image_free(&image);
        return false;
    }

    png_image_free(&image);
    return true;
}

bool writePNG(const std::string& path,
              const unsigned char* pixels,
              int width, int height) {

    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_GRAY;

    if (!png_image_write_to_file(
            &image, path.c_str(), 0, pixels, 0, nullptr)) {
        std::cerr << "Cannot save: " << path << std::endl;
        return false;
    }

    return true;
}

void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

int main(int argc, char* argv[]) {

    std::string inputDir = "dataset/input_images";
    std::string outputDir = "output";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            inputDir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[++i];
        } else {
            std::cout << "Usage: " << argv[0]
                      << " [--input directory] [--output directory]\n";
            return 1;
        }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << std::endl;

    createDirectory(outputDir);
    createDirectory(outputDir + "/grayscale");
    createDirectory(outputDir + "/blur");
    createDirectory(outputDir + "/edges");

    DIR* dir = opendir(inputDir.c_str());

    if (!dir) {
        std::cerr << "Cannot open input directory: "
                  << inputDir << std::endl;
        return 1;
    }

    std::vector<std::string> files;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;

        if (name.size() >= 4 &&
            name.substr(name.size() - 4) == ".png") {
            files.push_back(name);
        }
    }

    closedir(dir);
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No PNG images found." << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    int processed = 0;

    const dim3 block(16, 16);

    for (const std::string& filename : files) {

        std::string inputPath = inputDir + "/" + filename;

        std::vector<unsigned char> hostRGB;

        int width, height;

        if (!readPNG(inputPath, hostRGB, width, height)) {
            return 1;
        }

        size_t pixels = (size_t)width * height;
        size_t rgbBytes = pixels * 3;
        size_t grayBytes = pixels;

        unsigned char *d_rgb, *d_gray, *d_blur, *d_edges;

        CUDA_CHECK(cudaMalloc(&d_rgb, rgbBytes));
        CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));
        CUDA_CHECK(cudaMalloc(&d_blur, grayBytes));
        CUDA_CHECK(cudaMalloc(&d_edges, grayBytes));

        CUDA_CHECK(cudaMemcpy(
            d_rgb, hostRGB.data(), rgbBytes,
            cudaMemcpyHostToDevice));

        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        grayscaleKernel<<<grid, block>>>(
            d_rgb, d_gray, width, height);

        gaussianKernel<<<grid, block>>>(
            d_gray, d_blur, width, height);

        sobelKernel<<<grid, block>>>(
            d_blur, d_edges, width, height);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<unsigned char> hostGray(pixels);
        std::vector<unsigned char> hostBlur(pixels);
        std::vector<unsigned char> hostEdges(pixels);

        CUDA_CHECK(cudaMemcpy(
            hostGray.data(), d_gray, grayBytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            hostBlur.data(), d_blur, grayBytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            hostEdges.data(), d_edges, grayBytes,
            cudaMemcpyDeviceToHost));

        bool saved =
            writePNG(outputDir + "/grayscale/" + filename,
                     hostGray.data(), width, height) &&
            writePNG(outputDir + "/blur/" + filename,
                     hostBlur.data(), width, height) &&
            writePNG(outputDir + "/edges/" + filename,
                     hostEdges.data(), width, height);

        cudaFree(d_rgb);
        cudaFree(d_gray);
        cudaFree(d_blur);
        cudaFree(d_edges);

        if (!saved) {
            return 1;
        }

        processed++;

        if (processed % 10 == 0) {
            std::cout << "Processed " << processed
                      << "/" << files.size() << " images\n";
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "\nProcessing completed successfully!\n";
    std::cout << "Total images: " << processed << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Total time: " << seconds << " seconds\n";
    std::cout << "Output folders: grayscale, blur, edges\n";

    return 0;
}
