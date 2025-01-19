#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

#define TILE_SIZE 16

// The CUDA kernel
__global__ void conv2d_kernel(float *input, float *filter, float *output, int H, int W, int N, int K)
{
  // Calculate thread id, block id, etc.
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  // Declare shared memory
  __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];

  // Iterate over each image
  for (int n = 0; n < N; n++)
  {
    // Initialize shared memory
    for (int i = ty; i < TILE_SIZE + 2; i += blockDim.y)
    {
      for (int j = tx; j < TILE_SIZE + 2; j += blockDim.x)
      {
        int y = by * blockDim.y + i - 1;
        int x = bx * blockDim.x + j - 1;
        if (y >= 0 && y < H && x >= 0 && x < W)
        {
          tile[i][j] = input[n * H * W + y * W + x]; // Adjusted to handle multiple images
        }
        else
        {
          tile[i][j] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Perform convolution
    if (row < H && col < W)
    {
      for (int k = 0; k < K; k++)
      {
        float sum = 0.0f;
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            sum += tile[ty + i][tx + j] * filter[k * 3 * 3 + i * 3 + j];
          }
        }
        output[n * K * H * W + k * H * W + row * W + col] = sum; // Adjusted to handle multiple images
      }
    }
    __syncthreads();
  }
}

int main(int argc, char *argv[])
{
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

    // Allocate/move data using cudaMalloc and cudaMemCpy

    float *d_input = NULL;
    float *d_filter = NULL;
    float *d_output = NULL;

    cudaMalloc((void**)&d_input, N * inputHeight * inputWidth * sizeof(float));
    cudaMalloc((void**)&d_filter, K * filterHeight * filterWidth * sizeof(float));
    cudaMalloc((void**)&d_output, N * K * inputHeight * inputWidth * sizeof(float));

    // Copy the input image and filter from host to device
    cudaMemcpy(d_input, inputImage, N * inputHeight * inputWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filterValues, K * filterHeight * filterWidth * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel

    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    // Calculate the grid size (number of blocks per grid)
    int gridWidth = (inputWidth + TILE_SIZE - 1) / TILE_SIZE;
    int gridHeight = (inputHeight + TILE_SIZE - 1) / TILE_SIZE;
    dim3 gridDim(gridWidth, gridHeight, N);

    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, inputHeight, inputWidth, N, K);

    // Allocate memory for the output on the CPU
    float *h_output = (float *)malloc(N * K * inputHeight * inputWidth * sizeof(float));

    // Copy the output back to the CPU
    cudaMemcpy(h_output, d_output, N * K * inputHeight * inputWidth * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    for (int n = 0; n < N; n++)
    {
      for (int k = 0; k < K; k++)
      {
        for (int i = 0; i < inputHeight; i++)
        {
          for (int j = 0; j < inputWidth; j++)
          {
            std::cout << std::fixed << std::setprecision(3) << h_output[n * K * inputHeight * inputWidth + k * inputHeight * inputWidth + i * inputWidth + j] << " ";
          }
          std::cout << "\n";
        }

      }

    }

    // Clean up the memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    free(inputImage);
    free(filterValues);
    free(h_output);

    return 0;
}
