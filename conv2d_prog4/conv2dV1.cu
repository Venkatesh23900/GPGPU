#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

// The CUDA kernel
__global__ void conv2d_kernel(float *input, float *filter, float *output, int H, int W, int N, int K)
{
  // Calculate the global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure we don't go out of bounds
  if (tid < N * K * (H - 2) * (W - 2))
  {
    int n = tid / (K * (H - 2) * (W - 2));                         // Determine the image number
    int k = (tid % (K * (H - 2) * (W - 2))) / ((H - 2) * (W - 2)); // Determine the filter number
    int h = ((tid % ((H - 2) * (W - 2))) / (W - 2)) + 1;           // Determine the row in the image
    int w = (tid % (W - 2)) + 1;                                   // Determine the column in the image

    // Perform the convolution operation for this pixel
    float sum = 0;
    for (int i = -1; i <= 1; i++)
    {
      for (int j = -1; j <= 1; j++)
      {
        sum += input[n * H * W + (h + i) * W + (w + j)] * filter[k * 3 * 3 + (i + 1) * 3 + (j + 1)];
      }
    }

    // Write the result to the output array
    output[tid] = sum;
  }
}

int main(int argc, char *argv[]) {

    // Read the inputs from command line

    // Exit with an error if the number of command-line arguments is incorrect.
    if (argc != 3)
    {
        printf("Error: Expected 2 command-line arguments but was provided %d.\n", (argc - 1));
        exit(EXIT_FAILURE);
    }

    FILE* fp_input = NULL;  // Pointer to input image file
    FILE* fp_filter = NULL; // Pointer to filter image file

    char* input = NULL;
    char* filter = NULL;

    input = argv[1];
    filter = argv[2];

    fp_input = fopen(input, "r");
    fp_filter = fopen(filter, "r");

    // Check if the input files can be opened
    
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

    int inputHeight{}, inputWidth{}, N{};
    int filterHeight{}, filterWidth, K{};


    // Reading height, width of input image
    fscanf(fp_input, "%d", &inputHeight);
    fscanf(fp_input, "%d", &inputWidth);
    fscanf(fp_input, "%d", &N);

    // Reading height of the filter image
    fscanf(fp_filter, "%d", &filterHeight);
    fscanf(fp_filter, "%d", &K);
    filterWidth = filterHeight;

    // std::cout << inputHeight << "\n";
    // std::cout << inputWidth << "\n";
    // std::cout << N << "\n";


    // std::cout << filterHeight << "\n";
    // std::cout << K << "\n";

    // Allocate memory for the input image and filter
    float* inputImage = (float*) malloc(N * inputHeight * inputWidth * sizeof(float));
    float* filterValues = (float*) malloc(K * filterHeight * filterWidth * sizeof(float)); 

    // Read the input image values
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                fscanf(fp_input, "%f", &inputImage[n * inputHeight * inputWidth + i * inputWidth + j]);
            }
        }
    }

    // Read the filter values
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < filterHeight; i++) {
            for (int j = 0; j < filterWidth; j++) {
                fscanf(fp_filter, "%f", &filterValues[k * filterHeight * filterWidth + i * filterWidth + j]);
            }
        }
    }

    // Closing files

    fclose(fp_input);
    fclose(fp_filter);

    // // Print input image
    // for (int n = 0; n < N; n++)
    // {
    //     for (int i = 0; i < inputHeight; i++)
    //     {
    //         for (int j = 0; j < inputWidth; j++)
    //         {
    //             std::cout << std::fixed << std::setprecision(3) << inputImage[n * inputHeight * inputWidth + i * inputWidth + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // // Print filter values
    // for (int k = 0; k < K; k++)
    // {
    //     for (int i = 0; i < filterHeight; i++)
    //     {
    //         for (int j = 0; j < filterWidth; j++)
    //         {
    //             std::cout << std::fixed << std::setprecision(3) << filterValues[k * filterHeight * filterWidth + i * filterWidth + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // Padding
    int padding = 1;
    int paddedHeight = inputHeight + 2 * padding;
    int paddedWidth = inputWidth + 2 * padding;

    // Allocate memory for the padded image
    float *paddedImage = (float *)malloc(N * paddedHeight * paddedWidth * sizeof(float));

    // Initialize the padded images to zero
    for (int n = 0; n < N; n++)
    {
      for (int i = 0; i < paddedHeight; i++)
      {
        for (int j = 0; j < paddedWidth; j++)
        {
          paddedImage[n * paddedHeight * paddedWidth + i * paddedWidth + j] = 0.0f;
        }
      }
    }

    // Copy the original image into the center of the padded image
    for (int n = 0; n < N; n++)
    {
      for (int i = 0; i < inputHeight; i++)
      {
        for (int j = 0; j < inputWidth; j++)
        {
          paddedImage[n * paddedHeight * paddedWidth + (i + padding) * paddedWidth + (j + padding)] = inputImage[n * inputHeight * inputWidth + i * inputWidth + j];
        }
      }
    }

    // // Print padded images
    // for (int n = 0; n < N; n++)
    // {
    //     for (int i = 0; i < paddedHeight; i++)
    //     {
    //         for (int j = 0; j < paddedWidth; j++)
    //         {
    //             std::cout << std::fixed << std::setprecision(3) << paddedImage[n * paddedHeight * paddedWidth + i * paddedWidth + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // Allocate/move data using cudaMalloc and cudaMemCpy

    float *d_padded = NULL;
    float *d_filter = NULL;
    float *d_output = NULL;

    cudaMalloc((void**)&d_padded, N * paddedHeight * paddedWidth * sizeof(float));
    cudaMalloc((void**)&d_filter, K * filterHeight * filterWidth * sizeof(float));
    cudaMalloc((void**)&d_output, N * K * (paddedHeight - 2) * (paddedWidth - 2) * sizeof(float));

    // Copy the input image and filter from host to device
    cudaMemcpy(d_padded, paddedImage, N * paddedHeight * paddedWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filterValues, K * filterHeight * filterWidth * sizeof(float), cudaMemcpyHostToDevice);


    // Launch the kernel

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * K * (paddedHeight - 2) * (paddedWidth - 2) + threadsPerBlock - 1) / threadsPerBlock;
    conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_padded, d_filter, d_output, paddedHeight, paddedWidth, N, K);
    
    // Print the output

    // Copy the output data from device to host
    float* h_output = (float*) malloc(N * K * (paddedHeight - 2) * (paddedWidth - 2) * sizeof(float));
    cudaMemcpy(h_output, d_output, N * K * (paddedHeight - 2) * (paddedWidth - 2) * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output

    for (int n = 0; n < N; n++)
    {
      for (int k = 0; k < K; k++)
      {
        for (int i = 0; i < paddedHeight - 2; i++)
        {
          for (int j = 0; j < paddedWidth - 2; j++)
          {
            std::cout << std::fixed << std::setprecision(3) << h_output[n * K * (paddedHeight - 2) * (paddedWidth - 2) + k * (paddedHeight - 2) * (paddedWidth - 2) + i * (paddedWidth - 2) + j] << " ";
          }
          std::cout << "\n";
        }
      }
    }

    // Clean up the memory
    cudaFree(d_padded);
    cudaFree(d_filter);
    cudaFree(d_output);
    free(inputImage);
    free(filterValues);
    free(paddedImage);
    free(h_output);

    return 0;
}
