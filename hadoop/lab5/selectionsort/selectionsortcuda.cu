#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_functions.h>

void _CheckCudaError(const cudaError_t cudaError, const char* file, const int line)
{
    if (cudaError != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << cudaGetErrorString(cudaError) << " (" << file << ":" << line << ")\n";
        exit(EXIT_FAILURE);
    }
}
#define CheckCudaError(call) _CheckCudaError((call), __FILE__, __LINE__)

#define N       1024
#define blkSize 1024
#define grdSize 1

__global__ void init(unsigned long long seed, curandState_t* states)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void build(int* d_data, curandState_t* states)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[id] = ceilf(curand_uniform(&states[id]) * N);
}

// Finds the maximum element within a warp and gives the maximum element to
// thread with lane id 0. Note that other elements do not get lost but their
// positions are shuffled.
__inline__ __device__ int warpMax(int data, unsigned int threadId)
{
    for (int mask = 16; mask > 0; mask /= 2) {
        int dual_data = __shfl_xor(data, mask, 32);
        if (threadId & mask)
            data = min(data, dual_data);
        else
            data = max(data, dual_data);
    }
    return data;
}

__global__ void selection32(int* d_data, int* d_data_sorted)
{
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int laneId = threadIdx.x % 32;

    int n = N;
    while(n-- > 0) {
        // get the maximum element among d_data and put it in d_data_sorted[n]
        int data = d_data[threadId];
        data = warpMax(data, threadId);
        d_data[threadId] = data;

        // now maximum element is in d_data[0]
        if (laneId == 0) {
            d_data_sorted[n] = d_data[0];
            d_data[0] = INT_MIN; // this element is ignored from now on
        }
    }
}

void swap(int *xp, int *yp) 
{ 
    int temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 
  
void selection_sort_seq(int* arr, int n) 
{ 
    int i, j, min_idx; 
  
    // One by one move boundary of unsorted subarray 
    for (i = 0; i < n-1; i++) 
    { 
        // Find the minimum element in unsorted array 
        min_idx = i; 
        for (j = i+1; j < n; j++) 
        if (arr[j] < arr[min_idx]) 
            min_idx = j; 
  
        // Swap the found minimum element with the first element 
        swap(&arr[min_idx], &arr[i]); 
    } 
} 


int main()
{
    int* d_data;
    int* d_data_sorted;
    int* h_data;
    int* h_data_sorted;
    curandState_t* states;

    // allocate host and device memory
    CheckCudaError(cudaMalloc(&d_data, sizeof(int) * N));
    CheckCudaError(cudaMalloc(&d_data_sorted, sizeof(int) * N));
    h_data = (int*)malloc(sizeof(int) * N);
    h_data_sorted = (int*)malloc(sizeof(int) * N);
    CheckCudaError(cudaMalloc(&states, sizeof(curandState_t) * N));

    // build random data
    init<<<grdSize, blkSize>>>(time(0), states);
    build<<<grdSize, blkSize>>>(d_data, states);

    // print random data
    CheckCudaError(cudaMemcpy(h_data, d_data, sizeof(int) * N, cudaMemcpyDeviceToHost));
    printf("Testing the performance of selection sort on CUDA \nand comparing it with CPU running time\n");
    printf("N: %d\n", N);
    
    printf("Running Sequential Selection Sort on CPU\n");
    printf("\n");

    // sequential selection sort
    int* d_data_seq = (int*)malloc(sizeof(int) * N);
    memcpy(d_data_seq, h_data, sizeof(int) * N);
    
    clock_t start, end;
    double time_used;

    start = clock();
    selection_sort_seq(d_data_seq, N);
    end = clock();

    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Sequential selection sort time: %f\n", time_used);


    printf("\nRunning Selection Sort on GPU\n");
    // parallel selection-sort
    start = clock();
    selection32<<<grdSize, blkSize>>>(d_data, d_data_sorted);
    end = clock();

    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel selection sort time: %f\n", time_used);
    // print sorted data
    CheckCudaError(cudaMemcpy(h_data_sorted, d_data_sorted, sizeof(int) * N, cudaMemcpyDeviceToHost));


    // free allocated memory
    CheckCudaError(cudaFree(d_data));
    CheckCudaError(cudaFree(d_data_sorted));
    free(h_data);
    free(h_data_sorted);
    CheckCudaError(cudaFree(states));

    return 0;
}