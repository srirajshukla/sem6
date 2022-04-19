#include <cstdlib>
#include <iostream>

__global__ void matrixMul(int *a, int *b, int *c, int N){
    // calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // check if the thread is valid
    if (row < N && col < N) {
        // calculate the value of the element
        int sum = 0;
        for(int i=0; i<N; i++){
            sum += a[row * N + i] * b[i * N + col];
        }

        // store the value in the result matrix
        c[row * N + col] = sum;
    }
}

// initialize a square matrix between 1 and X
void init_matrix(int *m, int N, int x) {
    for (int i=0; i<N*N; i++) {
        m[i] = rand() % x + 1;
    }
}

// verify the result on cpu
void verify_matrix(int *a, int *b, int *c, int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            int sum = 0;
            for (int k=0; k<N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            if (sum != c[i * N + j]) {
                printf("Error: (%d, %d) = %d, should be %d\n", i, j, c[i * N + j], sum);
            }
        }
    }
}

int main(int argc, char **argv) {
    // the size of the matrix
    int N=8;
    // number of elements in between 1 and x
    int x = 8;
    // number of blocks (cpu)
    int blck_size = 1;

    // square matrix dimension (N*N)
    int bytes = N*N*sizeof(int);


    float gpu_elapsed_time, cpu_elapsed_time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory for matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize matrices
    init_matrix(a, N, x);
    init_matrix(a, N, x);

    // set our block and grid dimensions
    int threads = 1*1;
    int blocks = blck_size; // (N + threads - 1) / threads;
    std:: cout << "\n";
    std:: cout << "Matrix size: " << N << "*" << N << std::endl;
    std:: cout << "Block size: " << blck_size << std::endl;
    std:: cout << "Threads: " << threads << "\n\n";


    // setup kernel parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    cudaEventRecord(start, 0);

    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time, start, stop);

    std::cout << "GPU Elapsed Time: " << gpu_elapsed_time << " ms\n";

    cudaEventRecord(start, 0);
    verify_matrix(a, b, c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time, start, stop);

    std::cout << "CPU Elapsed Time: " << cpu_elapsed_time << " ms\n";

    std::cout << "Program Completed Successfully\n\n";
}