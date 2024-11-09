#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The CUDA kernel for 2D convolution

__global__ void conv2d_kernel(float* paddedInput, float* filter, float* output, int paddedHeight, int paddedWidth, int filterHeight, int filterWidth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < paddedHeight - filterHeight + 1 && col < paddedWidth - filterWidth + 1) {
        float result = 0.0;

        // Apply the filter to the padded image
        for (int i = 0; i < filterHeight; ++i) {
            for (int j = 0; j < filterWidth; ++j) {
                int inputRow = row + i;
                int inputCol = col + j;
                result += paddedInput[inputRow * paddedWidth + inputCol] * filter[i * filterWidth + j];
            }
        }

        // Store the result in the output image matrix
        output[row * (paddedWidth - filterWidth + 1) + col] = result;
    }
}


int main(int argc, char *argv[]) {

    // Read the inputs from command line

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Create CUDA events

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Exit with an error if the number of command-line arguments is incorrect.
    if (argc != 3)
    {
        printf("Error: Expected 2 command-line arguments but was provided %d.\n", (argc - 1));
        exit(EXIT_FAILURE);
    }

    FILE* fp_input;  // pointer to input image file
    FILE* fp_filter; // pointer to filter image file

    char* input;
    char* filter;

    input = argv[1];
    filter = argv[2];

    fp_input = fopen(input, "r");
    fp_filter = fopen(filter, "r");

    if (fp_input == (FILE *) NULL)
    {
        // Exit with an error if file open failed.
        printf("Error: Unable to open file %s\n", input);
        exit(EXIT_FAILURE);
    }

    if (fp_filter == (FILE *) NULL)
    {
        // Exit with an error if file open failed.
        printf("Error: Unable to open file %s\n", filter);
        exit(EXIT_FAILURE);
    }

    int inputHeight, inputWidth;
    int filterHeight, filterWidth;


    // reading height, width of input image
    fscanf(fp_input, "%d", &inputHeight);
    fscanf(fp_input, "%d", &inputWidth);

    // reading height of the filter image
    fscanf(fp_filter, "%d", &filterHeight);
    filterWidth = filterHeight;

    //std::cout << "input image dimensions: " << inputHeight << " x " << inputWidth << "\n";
    //std::cout << "filter image dimensions: " << filterHeight << " x " << filterWidth << "\n";

    // store input image
    float* input_image = new float[inputHeight*inputWidth];

    for(int i=0; i<inputHeight; i++)
    {   
        for(int j=0; j<inputWidth; j++)
        {
            fscanf(fp_input, "%f", &input_image[i*inputWidth + j]);
        }
    }

    // store filter image
    float* filter_image = new float[filterHeight*filterWidth];

    for(int i=0; i<filterHeight; i++)
    {   
        for(int j=0; j<filterWidth; j++)
        {
            fscanf(fp_filter, "%f", &filter_image[i*filterWidth + j]);
        }
    }

    // store padded image

    int padding = (filterHeight - 1)/2;

    int paddedHeight = inputHeight + 2*padding;
    int paddedWidth = inputWidth + 2*padding;

    float* padded_image = new float[paddedHeight*paddedWidth];

    // initializing the padded image

    for(int i=0; i<paddedHeight; i++)
    {
        for(int j=0; j<paddedWidth; j++)
        {
            padded_image[i*paddedWidth + j] = 0.0000;
        }
    }

    // std::cout << "padded image dimensions: " << paddedHeight << " x " << paddedWidth << "\n";

    // padded image

    // Copy the input image to the center of the padded image
    
    for (int i = 0; i < inputHeight; ++i)
    {
        for (int j = 0; j < inputWidth; ++j)
        {
            padded_image[(i + padding) * paddedWidth + (j + padding)] = input_image[i * inputWidth + j];
        }
    }

    // // print padded image

    // for (int i = 0; i < paddedHeight; i++)
    // {
    //     for (int j = 0; j < paddedWidth; j++)
    //     {
    //         std::cout << padded_image[i*paddedWidth + j] << " ";
    //     }
        
    //     std::cout << "\n";

    // }

    // // print filter image

    // for (int i = 0; i < filterHeight; i++)
    // {
    //     for (int j = 0; j < filterWidth; j++)
    //     {
    //         std::cout << filter_image[i*filterWidth + j] << " ";
    //     }
        
    //     std::cout << "\n";

    // }

    // // print input image

    // for (int i = 0; i < inputHeight; i++)
    // {
    //     for (int j = 0; j < inputWidth; j++)
    //     {
    //         std::cout << input_image[i*inputWidth + j] << " ";
    //     }
        
    //     std::cout << "\n";

    // }


    // Allocate/move data using cudaMalloc and cudaMemCpy

    float *d_padded = NULL;
    float *d_filter = NULL;
    float *d_output = NULL;

    err = cudaMalloc((void**)&d_padded, paddedHeight*paddedWidth*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device padded image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_filter, filterHeight*filterWidth*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device filter image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_output, inputHeight*inputWidth*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // copy data from host to device

    err = cudaMemcpy(d_padded, padded_image, paddedHeight * paddedWidth * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy padded image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_filter, filter_image, filterHeight * filterWidth * sizeof(float), cudaMemcpyHostToDevice);

     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy filter image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the kernel

    dim3 blockDim(16,16); // 256 threads/block
    int gridX = (paddedWidth + blockDim.x - 1) / blockDim.x;
    int gridY = (paddedHeight + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridX, gridY); // thread blocks/grid


    cudaEventRecord(start);
    conv2d_kernel<<<gridDim, blockDim>>>(d_padded, d_filter, d_output, paddedHeight, paddedWidth, filterHeight, filterWidth);
    cudaEventRecord(stop);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch conv2d kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    // Print the output

    // allocate memory for convoluted image in host

    float *output_image = new float[inputHeight*inputWidth];

    // copy the output image from device to host

    err = cudaMemcpy(output_image, d_output, inputHeight * inputWidth * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output image from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the kernel execution time

    //std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";


    // std::cout << "Output image dimensions: " << inputHeight << " x " << inputWidth << "\n";

    for (int i = 0; i < inputHeight; ++i)
    {
        for (int j = 0; j < inputWidth; ++j)
        {
            std::cout << std::fixed << std::setprecision(3) << output_image[i * inputWidth + j] << "\n";
        }

    }

    // clean host memory

    delete[] input_image;
    
    delete[] filter_image;

    delete[] padded_image;

    delete[] output_image;


    // clean device memory

    cudaFree(d_padded);

    cudaFree(d_filter);
    
    cudaFree(d_output);
    
    cudaFree(d_output);

    // close file

    fclose(fp_input);
    fclose(fp_filter);

    return 0;
}
