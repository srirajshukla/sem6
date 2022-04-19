#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

static const int blockSize = 2047; // array size
static const int iterations = 10;  // number of iterations
int countBlocks = 2500;

inline void cudaCheck(const cudaError_t &err, const std::string &mes)
{
    if (err != cudaSuccess)
    {
        std::cout << (mes + " - " + cudaGetErrorString(err)) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__device__ void swap(int *a, int *b)
{
    const int t = *a;
    *a = *b;
    *b = t;
}

__device__ void maxHeapify(int *maxHeap, int heapSize, int idx)
{
    int largest = idx;          // Initialize largest as root
    int left = (idx << 1) + 1;  // left = 2*idx + 1
    int right = (idx + 1) << 1; // right = 2*idx + 2

    // See if left child of root exists and is greater than root
    if (left < heapSize && maxHeap[left] > maxHeap[largest])
    {
        largest = left;
    }

    // See if right child of root exists and is greater than
    // the largest so far
    if (right < heapSize && maxHeap[right] > maxHeap[largest])
    {
        largest = right;
    }

    // Change root, if needed
    if (largest != idx)
    {
        swap(&maxHeap[largest], &maxHeap[idx]);
        maxHeapify(maxHeap, heapSize, largest);
    }
}

// A utility function to create a max heap of given capacity
__device__ void createAndBuildHeap(int *array, int size)
{
    // Start from bottommost and rightmost internal mode and heapify all
    // internal modes in bottom up way
    for (int i = (size - 2) / 2; i >= 0; --i)
    {
        maxHeapify(array, size, i);
    }
}

__global__ void heapSortKernel(int *iA, int size)
{
    // A = A + blockIdx.x * blockSize;
    iA = iA + blockIdx.x * blockSize;
    __shared__ int A[blockSize];
    for (int i = threadIdx.x; i < blockSize; i += blockDim.x)
    {
        A[i] = iA[i];
    }
    __syncthreads();
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIdx.x == 0)
    {
        // Build a heap from the input data.
        createAndBuildHeap(A, size);

        // Repeat following steps while heap size is greater than 1.
        // The last element in max heap will be the minimum element
        int changedSizeOfHeap = size;
        while (changedSizeOfHeap > 1)
        {
            // The largest item in Heap is stored at the root. Replace
            // it with the last item of the heap followed by reducing the
            // size of heap by 1.
            swap(A, &A[changedSizeOfHeap - 1]);
            --changedSizeOfHeap; // Reduce heap size

            // Finally, heapify the root of tree.
            maxHeapify(A, changedSizeOfHeap, 0);
        }
    }
    for (int i = threadIdx.x; i < blockSize; i += blockDim.x)
    {
        iA[i] = A[i];
    }
}

void heapifySeq(int arr[], int n, int i)
{
    int largest = i;   // Initialize largest as root
    int l = 2 * i + 1; // left = 2*i + 1
    int r = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root
    if (largest != i)
    {
        std::swap(arr[i], arr[largest]);

        // Recursively heapify the affected sub-tree
        heapifySeq(arr, n, largest);
    }
}

// main function to do heap sort
void heapSortSeq(int arr[], int n)
{
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapifySeq(arr, n, i);

    // One by one extract an element from heap
    for (int i = n - 1; i > 0; i--)
    {
        // Move current root to end
        std::swap(arr[0], arr[i]);

        // call max heapify on the reduced heap
        heapifySeq(arr, i, 0);
    }
}

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
    // Print the vector length to be used, and compute its size
    int numElements = blockSize * countBlocks;
    size_t size = numElements * sizeof(int);

    // Allocate the host input vector A
    int *h_A = (int *)malloc(size);
    int *h_A_seq = (int *)malloc(size);
    if (h_A == NULL)
    {
        std::cout << "Failed to allocate host vectors!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Allocate the device input vector A
    int *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    cudaCheck(err, "failed to allocated device vector A");

    // Time timer;
    clock_t start, end;
    // timer.begin("common");
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        // h_A[i] = rand()/(int)RAND_MAX;
        h_A[i] = rand() % 1000 + 1;
        h_A_seq[i] = h_A[i];
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaCheck(err, "failed to copy vector A to device");

    std::cout << "Testing the performance of heap sort on CUDA \n and "
              << "comparing it with CPU running time" << std::endl;

    std::cout << "N = " << numElements << std::endl;

    std::cout << "Running on CPU\n";
    start = clock();
    heapSortSeq(h_A_seq, numElements);
    end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "CPU time taken = " << time_taken << " sec" << std::endl;

    // Launch the CUDA Kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = countBlocks;
    // timer.begin("sort");
    std::cout << "Running on GPU\n";
    start = clock();
    heapSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, blockSize);
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "GPU time taken = " << time_taken << " sec" << std::endl;

    cudaDeviceSynchronize();
    // timer.end("sort");
    err = cudaGetLastError();
    cudaCheck(err, "failed to launch kernel");

    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaCheck(err, "failed to copy vector A to host");

    int running_time = ((int)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "running time = " << running_time << std::endl;

    // Free device global memory
    err = cudaFree(d_A);
    cudaCheck(err, "failed to free device vector A");
    // Free host memory
    free(h_A);
    err = cudaDeviceReset();
    cudaCheck(err, "failed to deinitialize the device");

    std::cout << "Done." << std::endl;
    return 0;
}